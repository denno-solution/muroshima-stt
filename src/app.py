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

from models import AudioTranscription, get_db
from stt_wrapper import STTModelWrapper
from text_structurer import TextStructurer
from env_watcher import check_env_changes, display_env_status
from app_settings import AppSettings
from auth import check_password, logout
from semantic_search import get_semantic_search_engine
from rag_qa import get_rag_qa_system

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
                stt_wrapper = STTModelWrapper(selected_model)
                text_structurer = TextStructurer() if use_structuring else None
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
                    logger.info(f"文字起こし実行中: {uploaded_file.name} (モデル: {selected_model})")
                    transcription = stt_wrapper.transcribe(tmp_path)
                    
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
                            
                            # ベクトルDBに追加
                            try:
                                search_engine = get_semantic_search_engine()
                                doc_id = f"audio_{audio_record.音声ID}"
                                metadata = {
                                    "audio_id": audio_record.音声ID,
                                    "file_path": uploaded_file.name,
                                    "recording_time": audio_record.録音時刻.isoformat(),
                                    "duration": duration,
                                    "speakers": 1,
                                    "tags": tags or ""
                                }
                                if structured_data:
                                    metadata["structured_data"] = str(structured_data)
                                
                                search_engine.add_document(doc_id, transcription, metadata)
                                logger.info(f"Added to vector DB: {doc_id}")
                            except Exception as e:
                                logger.warning(f"Failed to add to vector DB: {str(e)}")
                                
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
        # 新しい録音があれば保存
        if audio_bytes != st.session_state.mic_audio_bytes:
            st.session_state.mic_audio_bytes = audio_bytes
            st.session_state.mic_processing = False
        
        st.success("録音完了！")
        
        # 確認ダイアログ
        if not st.session_state.mic_processing:
            if st.button("🚀 文字起こしてデータベースに保存しますか？", type="primary", key="mic_process_button"):
                st.session_state.mic_processing = True
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
                stt_wrapper = STTModelWrapper(selected_model)
                text_structurer = TextStructurer() if use_structuring else None
                
                # 文字起こし実行
                with st.spinner("文字起こし中..."):
                    transcription = stt_wrapper.transcribe(tmp_path)
                    
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
                            
                            # ベクトルDBに追加
                            try:
                                search_engine = get_semantic_search_engine()
                                doc_id = f"audio_{audio_record.音声ID}"
                                metadata = {
                                    "audio_id": audio_record.音声ID,
                                    "file_path": result["ファイル名"],
                                    "recording_time": timestamp.isoformat(),
                                    "duration": duration,
                                    "speakers": 1,
                                    "tags": tags or ""
                                }
                                if structured_data:
                                    metadata["structured_data"] = str(structured_data)
                                
                                search_engine.add_document(doc_id, transcription, metadata)
                                logger.info(f"Added to vector DB: {doc_id}")
                            except Exception as e:
                                logger.warning(f"Failed to add to vector DB: {str(e)}")
                                
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
    st.markdown("- 録音ボタンを押してから話してください")
    st.markdown("- 録音終了後、「文字起こして保存」ボタンをクリック")
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
    
    # セマンティック検索機能
    st.subheader("🔍 セマンティック検索")
    
    # 検索タブ
    search_tab1, search_tab2, search_tab3 = st.tabs(["🤖 AI質問応答", "💭 意味検索", "📋 全レコード"])
    
    with search_tab1:
        st.subheader("🤖 AI質問応答")
        st.markdown("**議事録に関する質問を自然言語で入力してください。AIが関連する情報を検索して回答します。**")
        
        # 質問入力
        question = st.text_area(
            "質問を入力してください",
            placeholder="例：\n・予算削減についてこれまでどのような議論がありましたか？\n・人事制度の見直しで決まったことを教えてください\n・プロジェクトの進捗状況はどうなっていますか？",
            height=100,
            help="具体的で明確な質問ほど、正確な回答が得られます。"
        )
        
        # 設定オプション
        col1, col2, col3 = st.columns(3)
        with col1:
            max_sources = st.selectbox("参照する記録数", [3, 5, 8, 10], index=1)
        with col2:
            min_similarity = st.slider("類似度閾値", 0.3, 0.9, 0.5, 0.1)
        with col3:
            show_sources = st.checkbox("参照ソースを表示", value=True)
        
        if question and question.strip():
            try:
                # RAG質問応答システムを取得
                rag_system = get_rag_qa_system()
                
                # 質問応答実行
                with st.spinner("🔍 関連情報を検索してAIが回答を生成中..."):
                    result = rag_system.answer_question(
                        question=question.strip(),
                        max_context_docs=max_sources,
                        min_similarity=min_similarity
                    )
                
                # 回答表示
                st.markdown("### 💬 AI回答")
                
                # 信頼度に応じてスタイル変更
                confidence = result.get('confidence', 0.0)
                if confidence >= 0.7:
                    st.success("🎯 高信頼度の回答")
                elif confidence >= 0.5:
                    st.info("📝 中程度の信頼度")
                else:
                    st.warning("⚠️ 低信頼度（参考程度）")
                
                # 回答本文
                st.markdown(result['answer'])
                
                # メタデータ表示
                metadata = result.get('metadata', {})
                col_meta1, col_meta2, col_meta3 = st.columns(3)
                with col_meta1:
                    st.metric("信頼度", f"{confidence:.1%}")
                with col_meta2:
                    st.metric("検索件数", metadata.get('search_count', 0))
                with col_meta3:
                    st.metric("参照記録", len(result.get('sources', [])))
                
                # 参照ソース表示
                if show_sources and result.get('sources'):
                    st.markdown("### 📚 参照した音声記録")
                    
                    for i, source in enumerate(result['sources'], 1):
                        with st.expander(f"📄 音声記録 {i} - ID {source.get('audio_id', 'Unknown')} (類似度: {source.get('similarity_score', 0):.3f})"):
                            col_src1, col_src2 = st.columns(2)
                            
                            with col_src1:
                                st.markdown(f"**📁 ファイル:** {source.get('file_path', 'N/A')}")
                                recording_time = source.get('recording_time', '')
                                if recording_time:
                                    st.markdown(f"**📅 録音時刻:** {recording_time[:19]}")
                            
                            with col_src2:
                                st.markdown(f"**🎯 類似度:** {source.get('similarity_score', 0):.3f}")
                            
                            st.markdown("**📝 内容:**")
                            st.write(source.get('excerpt', ''))
                
                # エラー情報がある場合は表示
                if 'error' in metadata:
                    st.error(f"エラー詳細: {metadata['error']}")
                    
            except Exception as e:
                st.error(f"質問応答エラー: {str(e)}")
                logger.error(f"RAG QA error: {str(e)}")
        
        elif question and not question.strip():
            st.info("質問を入力してください。")
        
        # 使い方のヒント
        st.markdown("---")
        st.markdown("**💡 使い方のコツ:**")
        st.markdown("- 具体的な質問ほど正確な回答が得られます")
        st.markdown("- 「いつ」「誰が」「何を」を含めると効果的です")
        st.markdown("- 複数のトピックを一度に聞くより、個別に質問してください")

    with search_tab2:
        st.subheader("💭 意味検索")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input(
                "検索したい内容を入力してください",
                placeholder="例: 予算削減について、人事制度の見直し、プロジェクトの進捗状況",
                help="キーワードだけでなく、文章でも検索できます。意味が似ている内容も自動で見つけます。"
            )
        
        with col2:
            search_limit = st.selectbox("表示件数", [5, 10, 20, 50], index=1, key="semantic_search_limit")
        
        if search_query:
            try:
                # セマンティック検索エンジンを取得
                search_engine = get_semantic_search_engine()
                
                # ベクトルDBの統計情報を表示
                stats = search_engine.get_collection_stats()
                if stats.get("total_documents", 0) == 0:
                    st.warning("⚠️ ベクトルDBにデータがありません。まず音声データを文字起こしして、ベクトルDBと同期してください。")
                    st.info("💡 音声を録音・アップロードすると、自動的にベクトルDBに追加されます。")
                else:
                    # 検索実行
                    with st.spinner("検索中..."):
                        results = search_engine.search(search_query, n_results=search_limit)
                    
                    if results:
                        st.success(f"🎯 {len(results)}件の結果が見つかりました")
                        
                        for i, result in enumerate(results, 1):
                            metadata = result['metadata']
                            similarity = result['similarity_score']
                            
                            # 結果を表示
                            with st.expander(f"#{i} 音声ID {metadata.get('audio_id', 'Unknown')} (類似度: {similarity:.3f})"):
                                st.markdown(f"**📄 文字起こしテキスト:**")
                                st.write(result['document'])
                                
                                col_meta1, col_meta2 = st.columns(2)
                                with col_meta1:
                                    st.markdown(f"**📁 ファイル:** {metadata.get('file_path', 'N/A')}")
                                    st.markdown(f"**🎙️ 発言人数:** {metadata.get('speakers', 'N/A')}")
                                
                                with col_meta2:
                                    recording_time = metadata.get('recording_time', '')
                                    if recording_time:
                                        st.markdown(f"**📅 録音時刻:** {recording_time[:19]}")  # ISO形式の日時から秒まで表示
                                    st.markdown(f"**⏱️ 録音時間:** {metadata.get('duration', 'N/A')}秒")
                                
                                # タグがある場合は表示
                                tags = metadata.get('tags', '')
                                if tags:
                                    st.markdown(f"**🏷️ タグ:** {tags}")
                    else:
                        st.info("該当する結果が見つかりませんでした。別のキーワードで試してみてください。")
                        
            except Exception as e:
                st.error(f"検索エラー: {str(e)}")
                logger.error(f"Semantic search error: {str(e)}")
    
    with search_tab3:
        st.subheader("📋 全レコード一覧")
        
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
            else:
                st.info("データベースにレコードがありません。")
        finally:
            db.close()

# フッター
st.divider()
st.markdown("🎙️ 音声文字起こしWebアプリ v1.0")