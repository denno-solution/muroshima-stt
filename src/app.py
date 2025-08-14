import streamlit as st
import pandas as pd
from pathlib import Path
import tempfile
import os
from datetime import datetime
import json
import librosa
import soundfile as sf
from dotenv import load_dotenv
import logging

# .envファイルを読み込む
load_dotenv()

from models import AudioTranscription, get_db, delete_record, delete_all_records
from stt_wrapper import STTModelWrapper
from text_structurer import TextStructurer
from env_watcher import check_env_changes, display_env_status
from app_settings import AppSettings
from auth import check_password, logout

# ロガーの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ログディレクトリの作成
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

# ファイルハンドラー
file_handler = logging.FileHandler(log_dir / "streamlit_app.log", encoding='utf-8')
file_handler.setLevel(logging.DEBUG)

# フォーマッター
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

# ページ設定
st.set_page_config(
    page_title="音声文字起こしWebアプリ",
    page_icon="🎙️",
    layout="wide"
)

# 環境変数の変更をチェック
check_env_changes()

# アプリ設定の初期化
settings = AppSettings()

# Basic認証チェック
if not check_password():
    st.stop()

# タイトル
st.title("🎙️ 音声文字起こしWebアプリ")
st.markdown("音声ファイルをアップロードして、文字起こしと構造化を行います。")

# セッション状態の初期化
if "transcriptions" not in st.session_state:
    st.session_state.transcriptions = []
if "processing" not in st.session_state:
    st.session_state.processing = False
if "mic_processing" not in st.session_state:
    st.session_state.mic_processing = False
if "mic_audio_bytes" not in st.session_state:
    st.session_state.mic_audio_bytes = None
if "settings" not in st.session_state:
    st.session_state.settings = settings

# サイドバー：設定
with st.sidebar:
    st.header("⚙️ 設定")
    
    # STTモデル選択
    available_models = STTModelWrapper.get_available_models()
    
    # 保存された選択を取得、なければデフォルト値を使用
    saved_model = settings.get_selected_stt_model()
    default_index = 4  # ElevenLabsをデフォルトに設定
    if saved_model and saved_model in available_models:
        default_index = available_models.index(saved_model)
    
    selected_model = st.selectbox(
        "STTモデルを選択",
        available_models,
        index=default_index,
        help="使用する音声認識モデルを選択してください"
    )
    
    # 選択が変更されたら保存
    if selected_model != saved_model:
        settings.set_selected_stt_model(selected_model)
        logger.info(f"STTモデルの選択を保存: {selected_model}")
    
    # 選択したモデルの要件チェック
    try:
        wrapper = STTModelWrapper(selected_model)
        requirements = wrapper.check_requirements()
        
        if requirements:
            st.subheader("環境変数の設定状況")
            for key, is_set in requirements.items():
                if is_set:
                    st.success(f"✅ {key}")
                else:
                    st.error(f"❌ {key} が設定されていません")
    except Exception as e:
        st.error(f"モデルの初期化エラー: {e}")
    
    st.divider()
    
    # 構造化設定
    st.subheader("構造化設定")
    use_structuring = st.checkbox(
        "Gemini Flash 2.5-liteで自動構造化", 
        value=settings.get_use_structuring()
    )
    
    # 設定が変更されたら保存
    if use_structuring != settings.get_use_structuring():
        settings.set_use_structuring(use_structuring)
        logger.info(f"構造化設定を保存: {use_structuring}")
    
    if use_structuring:
        gemini_key_set = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_AI_API_KEY"))
        if gemini_key_set:
            st.success("✅ Gemini API キー設定済み")
        else:
            st.error("❌ GEMINI_API_KEY が設定されていません")
    
    st.divider()
    
    # VAD設定
    st.subheader("🔇 VAD設定（コスト削減）")
    
    # VAD対応モデルかチェック
    vad_supported_models = STTModelWrapper.get_vad_supported_models()
    is_vad_supported = selected_model in vad_supported_models
    
    if is_vad_supported:
        enable_vad = st.checkbox(
            "VAD（無音除去）を有効化", 
            value=settings.get_vad_enabled(),
            help="無音部分を除去してSTT APIの処理時間とコストを削減します"
        )
        
        # 設定が変更されたら保存
        if enable_vad != settings.get_vad_enabled():
            settings.set_vad_enabled(enable_vad)
            logger.info(f"VAD設定を保存: {enable_vad}")
        
        if enable_vad:
            # VADパラメータの詳細設定
            with st.expander("🔧 VAD詳細設定"):
                vad_aggressiveness = st.slider(
                    "VAD厳しさ", 
                    min_value=0, 
                    max_value=3, 
                    value=settings.get_vad_aggressiveness(),
                    help="0=ゆるい（多くの音声を残す）、3=厳しい（明確な音声のみ残す）"
                )
                
                min_speech_ms = st.slider(
                    "最小スピーチ長 (ms)", 
                    min_value=100, 
                    max_value=1500, 
                    value=settings.get_vad_min_speech_ms(),
                    step=50,
                    help="この長さより短い音声は無視されます"
                )
                
                merge_gap_ms = st.slider(
                    "区間マージギャップ (ms)", 
                    min_value=50, 
                    max_value=1000, 
                    value=settings.get_vad_merge_gap_ms(),
                    step=50,
                    help="この長さ以下の無音は音声区間として統合されます"
                )
                
                # VADパラメータの保存
                if vad_aggressiveness != settings.get_vad_aggressiveness():
                    settings.set_vad_aggressiveness(vad_aggressiveness)
                if min_speech_ms != settings.get_vad_min_speech_ms():
                    settings.set_vad_min_speech_ms(min_speech_ms)
                if merge_gap_ms != settings.get_vad_merge_gap_ms():
                    settings.set_vad_merge_gap_ms(merge_gap_ms)
            
            st.success("💰 VADによりElevenLabsのコストを大幅削減できます")
    else:
        st.info(f"ℹ️ VADは {', '.join(vad_supported_models)} でサポートされています")
        enable_vad = False
    
    st.divider()
    
    # デバッグモード
    st.subheader("🐛 デバッグ設定")
    debug_mode = st.checkbox(
        "デバッグモードを有効化", 
        value=settings.get_debug_mode()
    )
    
    # 設定が変更されたら保存
    if debug_mode != settings.get_debug_mode():
        settings.set_debug_mode(debug_mode)
        logger.info(f"デバッグモード設定を保存: {debug_mode}")
    
    if debug_mode:
        # ログファイルの表示
        if st.button("📋 ログファイルを表示"):
            log_files = {
                "Streamlitログ": log_dir / "streamlit_app.log",
                "ElevenLabsデバッグログ": log_dir / "elevenlabs_debug.log"
            }
            
            for log_name, log_path in log_files.items():
                if log_path.exists():
                    st.subheader(f"📄 {log_name}")
                    try:
                        with open(log_path, "r", encoding="utf-8") as f:
                            # 最新の50行を表示
                            lines = f.readlines()
                            recent_lines = lines[-50:] if len(lines) > 50 else lines
                            st.code("".join(recent_lines), language="log")
                    except Exception as e:
                        st.error(f"ログ読み込みエラー: {e}")
                else:
                    st.info(f"{log_name}はまだ存在しません")
    
    st.divider()
    
    # 環境変数の状態表示
    display_env_status(sidebar=True)
    
    # ログアウトボタン（Basic認証が有効な場合のみ表示）
    if os.getenv("BASIC_AUTH_USERNAME") and os.getenv("BASIC_AUTH_PASSWORD"):
        st.divider()
        if st.button("🚪 ログアウト", type="secondary", use_container_width=True):
            logout()
            st.rerun()

# メインエリア
tab1, tab2, tab3, tab4 = st.tabs(["📤 アップロード", "🎙️ マイク録音", "📊 処理結果", "🗄️ データベース"])

with tab1:
    st.header("音声ファイルアップロード")
    
    # ファイルアップローダー
    uploaded_files = st.file_uploader(
        "音声ファイルを選択してください",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
        accept_multiple_files=True,
        help="複数ファイルを同時にアップロード可能です"
    )
    
    if uploaded_files:
        st.success(f"{len(uploaded_files)}個のファイルがアップロードされました")
        
        # アップロードファイルの情報表示
        file_info = []
        for file in uploaded_files:
            file_info.append({
                "ファイル名": file.name,
                "サイズ": f"{file.size / 1024:.1f} KB",
                "タイプ": file.type
            })
        
        df_files = pd.DataFrame(file_info)
        st.dataframe(df_files, use_container_width=True)
        
        # 処理開始ボタン
        if st.button("🚀 文字起こし開始", type="primary", use_container_width=True, disabled=st.session_state.processing):
            st.session_state.processing = True
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # STTモデルとテキスト構造化の初期化
            try:
                stt_wrapper = STTModelWrapper(selected_model, enable_vad=enable_vad)
                text_structurer = TextStructurer() if use_structuring else None
                
                # VADパラメータの準備
                vad_params = {}
                if enable_vad and stt_wrapper.is_vad_enabled():
                    vad_params = {
                        "min_speech_ms": min_speech_ms,
                        "merge_gap_ms": merge_gap_ms,
                        "vad_aggressiveness": vad_aggressiveness
                    }
                    logger.info(f"VAD有効: {vad_params}")
                
            except Exception as e:
                st.error(f"初期化エラー: {e}")
                st.session_state.processing = False
                st.stop()
            
            # 各ファイルを処理
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"処理中: {uploaded_file.name} ({idx + 1}/{len(uploaded_files)})")
                progress_bar.progress((idx + 1) / len(uploaded_files))
                
                try:
                    logger.info(f"処理開始: {uploaded_file.name}")
                    
                    # 一時ファイルとして保存
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    logger.debug(f"一時ファイル作成: {tmp_path}")
                    
                    # 音声ファイルの情報取得
                    audio_data, sr = librosa.load(tmp_path, sr=None)
                    duration = len(audio_data) / sr
                    logger.debug(f"音声ファイル情報: 時間={duration:.2f}秒, サンプリングレート={sr}Hz")
                    
                    # 文字起こし実行
                    logger.info(f"文字起こし実行中: {uploaded_file.name} (モデル: {selected_model}, VAD: {enable_vad})")
                    
                    # VAD対応の場合はメタデータも取得
                    if enable_vad and stt_wrapper.is_vad_enabled():
                        transcription, metadata = stt_wrapper.transcribe_with_metadata(tmp_path, vad_params)
                        if metadata.get('vad_stats'):
                            vad_stats = metadata['vad_stats']
                            logger.info(f"VAD統計: 元時間={vad_stats['original_duration_ms']}ms, "
                                       f"音声時間={vad_stats['speech_duration_ms']}ms, "
                                       f"圧縮率={vad_stats['compression_ratio']:.2%}")
                    else:
                        transcription = stt_wrapper.transcribe(tmp_path)
                        metadata = {}
                    
                    # エラーメッセージを含むタプルかチェック
                    error_msg = None
                    if isinstance(transcription, tuple) and transcription[0] is None:
                        error_msg = transcription[1]
                        transcription = None
                        logger.error(f"文字起こしエラー: {error_msg}")
                    
                    if transcription:
                        # 構造化処理
                        structured_data = None
                        tags = "未分類"
                        
                        if use_structuring and text_structurer:
                            structured_data = text_structurer.structure_text(transcription)
                            if structured_data:
                                tags = text_structurer.extract_tags(structured_data)
                        
                        # 結果を保存
                        result = {
                            "ファイル名": uploaded_file.name,
                            "録音時刻": datetime.now(),
                            "録音時間": duration,
                            "文字起こしテキスト": transcription,
                            "構造化データ": structured_data,
                            "タグ": tags,
                            "発言人数": 1  # デフォルト値
                        }
                        
                        st.session_state.transcriptions.append(result)
                        
                        # データベースに保存
                        db = next(get_db())
                        try:
                            audio_record = AudioTranscription(
                                音声ファイルpath=uploaded_file.name,
                                発言人数=1,
                                録音時刻=datetime.now(),
                                録音時間=duration,
                                文字起こしテキスト=transcription,
                                構造化データ=structured_data,
                                タグ=tags
                            )
                            db.add(audio_record)
                            db.commit()
                        finally:
                            db.close()
                    else:
                        # エラーメッセージがある場合は詳細を表示
                        if error_msg:
                            st.error(f"❌ {uploaded_file.name} の文字起こしに失敗しました")
                            st.error(f"エラー詳細: {error_msg}")
                            logger.error(f"文字起こし失敗: {uploaded_file.name}, エラー: {error_msg}")
                        else:
                            st.error(f"❌ {uploaded_file.name} の文字起こしに失敗しました（結果が空）")
                            logger.error(f"文字起こし失敗: {uploaded_file.name}, 結果が空")
                    
                    # 一時ファイルを削除
                    os.unlink(tmp_path)
                    logger.debug(f"一時ファイル削除: {tmp_path}")
                    
                except Exception as e:
                    error_msg = f"処理エラー ({uploaded_file.name}): {str(e)}"
                    st.error(error_msg)
                    logger.error(error_msg, exc_info=True)
            
            progress_bar.progress(1.0)
            status_text.text("✅ すべての処理が完了しました！")
            st.session_state.processing = False
            st.rerun()

with tab2:
    st.header("マイク録音")
    
    st.markdown("**マイクから直接音声を録音して文字起こしします**")
    
    # 録音機能
    audio_bytes = st.audio_input("🎙️ マイクで録音してください", help="録音ボタンを押して音声を録音し、停止ボタンで録音を終了してください")
    
    if audio_bytes:
        # 新しい録音があれば自動的に処理を開始
        if audio_bytes != st.session_state.mic_audio_bytes:
            st.session_state.mic_audio_bytes = audio_bytes
            st.session_state.mic_processing = True
            st.success("📁 録音完了！文字起こしを開始します...")
            st.rerun()
        
        # 録音データを処理
        if st.session_state.mic_processing:
            try:
                # 一時ファイルとして保存
                with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp_file:
                    # audio_bytesがUploadedFileオブジェクトの場合はgetvalue()でバイト列を取得
                    if hasattr(audio_bytes, 'getvalue'):
                        tmp_file.write(audio_bytes.getvalue())
                    else:
                        tmp_file.write(audio_bytes)
                    tmp_path = tmp_file.name
                
                logger.info(f"マイク録音処理開始: {tmp_path}")
                
                # 音声ファイルの情報取得
                try:
                    audio_data, sr = librosa.load(tmp_path, sr=None)
                    duration = len(audio_data) / sr
                    logger.debug(f"録音音声情報: 時間={duration:.2f}秒, サンプリングレート={sr}Hz")
                except Exception as e:
                    # librosaで読み込めない場合のフォールバック
                    duration = 0.0
                    logger.warning(f"音声情報取得失敗（処理は継続）: {e}")
                
                # STTモデルの初期化
                stt_wrapper = STTModelWrapper(selected_model, enable_vad=enable_vad)
                text_structurer = TextStructurer() if use_structuring else None
                
                # VADパラメータの準備
                vad_params = {}
                if enable_vad and stt_wrapper.is_vad_enabled():
                    vad_params = {
                        "min_speech_ms": min_speech_ms,
                        "merge_gap_ms": merge_gap_ms,
                        "vad_aggressiveness": vad_aggressiveness
                    }
                
                # 文字起こし実行
                with st.spinner("文字起こし中..."):
                    logger.info(f"マイク録音文字起こし実行中 (モデル: {selected_model}, VAD: {enable_vad})")
                    
                    # VAD対応の場合はメタデータも取得
                    if enable_vad and stt_wrapper.is_vad_enabled():
                        transcription, metadata = stt_wrapper.transcribe_with_metadata(tmp_path, vad_params)
                        if metadata.get('vad_stats'):
                            vad_stats = metadata['vad_stats']
                            logger.info(f"マイク録音VAD統計: 元時間={vad_stats['original_duration_ms']}ms, "
                                       f"音声時間={vad_stats['speech_duration_ms']}ms, "
                                       f"圧縮率={vad_stats['compression_ratio']:.2%}")
                    else:
                        transcription = stt_wrapper.transcribe(tmp_path)
                        metadata = {}
                    
                    # エラーメッセージを含むタプルかチェック
                    error_msg = None
                    if isinstance(transcription, tuple) and transcription[0] is None:
                        error_msg = transcription[1]
                        transcription = None
                        logger.error(f"マイク録音文字起こしエラー: {error_msg}")
                    
                    if transcription:
                        # 構造化処理
                        structured_data = None
                        tags = "マイク録音"
                        
                        if use_structuring and text_structurer:
                            with st.spinner("テキスト構造化中..."):
                                structured_data = text_structurer.structure_text(transcription)
                                if structured_data:
                                    tags = text_structurer.extract_tags(structured_data)
                        
                        # 結果を保存
                        timestamp = datetime.now()
                        result = {
                            "ファイル名": f"マイク録音_{timestamp.strftime('%Y%m%d_%H%M%S')}.webm",
                            "録音時刻": timestamp,
                            "録音時間": duration,
                            "文字起こしテキスト": transcription,
                            "構造化データ": structured_data,
                            "タグ": tags,
                            "発言人数": 1
                        }
                        
                        st.session_state.transcriptions.append(result)
                        
                        # データベースに保存
                        db = next(get_db())
                        try:
                            audio_record = AudioTranscription(
                                音声ファイルpath=result["ファイル名"],
                                発言人数=1,
                                録音時刻=timestamp,
                                録音時間=duration,
                                文字起こしテキスト=transcription,
                                構造化データ=structured_data,
                                タグ=tags
                            )
                            db.add(audio_record)
                            db.commit()
                            logger.info(f"マイク録音結果をデータベースに保存: {result['ファイル名']}")
                        finally:
                            db.close()
                        
                        # 結果表示
                        st.success("✅ 文字起こし完了！")
                        
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.subheader("文字起こし結果")
                            st.text_area("", transcription, height=200, key="mic_transcription")
                            st.write(f"**録音時間:** {duration:.1f}秒")
                            st.write(f"**タグ:** {tags}")
                        
                        with col2:
                            if structured_data:
                                st.subheader("構造化データ")
                                st.json(structured_data)
                            else:
                                st.info("構造化データはありません")
                    
                    else:
                        # エラーメッセージがある場合は詳細を表示
                        if error_msg:
                            st.error(f"❌ マイク録音の文字起こしに失敗しました")
                            st.error(f"エラー詳細: {error_msg}")
                        else:
                            st.error("❌ マイク録音の文字起こしに失敗しました（結果が空）")
                
                    # 一時ファイルを削除
                    os.unlink(tmp_path)
                    logger.debug(f"一時ファイル削除: {tmp_path}")
                    
                    # 処理完了後、状態をリセット
                    st.session_state.mic_processing = False
                    st.session_state.mic_audio_bytes = None
                    
            except Exception as e:
                error_msg = f"マイク録音処理エラー: {str(e)}"
                st.error(error_msg)
                logger.error(error_msg, exc_info=True)
                st.session_state.mic_processing = False
    
    st.divider()
    st.markdown("**💡 使い方のヒント:**")
    st.markdown("- 録音ボタンを押してから話し、停止ボタンを押して録音を終了してください")
    st.markdown("- 録音データは一時的に保存され、処理後に削除されます")

with tab3:
    st.header("処理結果")
    
    if st.session_state.transcriptions:
        for idx, result in enumerate(st.session_state.transcriptions):
            with st.expander(f"📁 {result['ファイル名']}", expanded=True):
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("基本情報")
                    st.write(f"**録音時刻:** {result['録音時刻'].strftime('%Y/%m/%d %H:%M')}")
                    st.write(f"**録音時間:** {result['録音時間']:.1f}秒")
                    st.write(f"**タグ:** {result['タグ']}")
                    
                    st.subheader("文字起こしテキスト")
                    st.text_area("", result['文字起こしテキスト'], height=200, key=f"text_{idx}")
                
                with col2:
                    if result.get('構造化データ'):
                        st.subheader("構造化データ")
                        st.json(result['構造化データ'])
                    else:
                        st.info("構造化データはありません")
    else:
        st.info("処理結果がありません。音声ファイルをアップロードして処理を開始してください。")

with tab4:
    st.header("データベース内容")
    
    # データベースから全レコードを取得
    db = next(get_db())
    try:
        records = db.query(AudioTranscription).all()
        
        if records:
            # データフレームに変換
            data = []
            for record in records:
                data.append({
                    "音声ID": record.音声ID,
                    "音声ファイル": record.音声ファイルpath,
                    "発言人数": record.発言人数,
                    "録音時刻": record.録音時刻,
                    "録音時間(s)": record.録音時間,
                    "タグ": record.タグ,
                    "文字起こし": record.文字起こしテキスト[:50] + "..." if len(record.文字起こしテキスト) > 50 else record.文字起こしテキスト
                })
            
            df = pd.DataFrame(data)
            
            # フィルタリング
            col1, col2 = st.columns([1, 1])
            with col1:
                tag_filter = st.selectbox("タグでフィルタ", ["すべて"] + list(df["タグ"].unique()))
            
            if tag_filter != "すべて":
                df = df[df["タグ"] == tag_filter]
            
            # データテーブル表示
            st.dataframe(df, use_container_width=True)
            
            # 詳細表示
            if st.checkbox("詳細を表示"):
                selected_id = st.selectbox("音声IDを選択", df["音声ID"].tolist())
                
                if selected_id:
                    record = db.query(AudioTranscription).filter_by(音声ID=selected_id).first()
                    if record:
                        st.subheader(f"音声ID: {record.音声ID} の詳細")
                        
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.write(f"**ファイル:** {record.音声ファイルpath}")
                            st.write(f"**録音時刻:** {record.録音時刻}")
                            st.write(f"**録音時間:** {record.録音時間}秒")
                            st.write(f"**タグ:** {record.タグ}")
                            
                            st.subheader("文字起こしテキスト")
                            st.text_area("", record.文字起こしテキスト, height=200)
                        
                        with col2:
                            if record.構造化データ:
                                st.subheader("構造化データ")
                                st.json(record.構造化データ)
                            
                            # 個別削除ボタン
                            st.divider()
                            st.subheader("🗑️ レコード削除")
                            if st.button(f"音声ID {record.音声ID} を削除", 
                                       type="secondary", 
                                       key=f"delete_{record.音声ID}",
                                       help="このレコードを完全に削除します"):
                                try:
                                    if delete_record(record.音声ID):
                                        st.success(f"音声ID {record.音声ID} を削除しました")
                                        st.rerun()
                                    else:
                                        st.error("削除に失敗しました")
                                except Exception as e:
                                    st.error(f"削除エラー: {str(e)}")
            
            # 一括削除機能
            st.divider()
            st.subheader("🗑️ 一括削除")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                if st.button("🚨 全レコード削除", 
                           type="secondary",
                           help="データベース内の全レコードを削除します"):
                    try:
                        deleted_count = delete_all_records()
                        st.success(f"{deleted_count}件のレコードを削除しました")
                        st.rerun()
                    except Exception as e:
                        st.error(f"一括削除エラー: {str(e)}")
            
            with col2:
                st.warning("⚠️ 削除したデータは復元できません。慎重に操作してください。")
        
        else:
            st.info("データベースにレコードがありません。")
    finally:
        db.close()

# フッター
st.divider()
st.markdown("🎙️ 音声文字起こしWebアプリ v1.0")