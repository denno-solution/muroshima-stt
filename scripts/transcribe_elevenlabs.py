import os
import json
from datetime import datetime
from pathlib import Path
from elevenlabs import ElevenLabs
import logging
from dotenv import load_dotenv

# .envファイルを読み込む
load_dotenv()

# ロガーの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# コンソールハンドラー
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# ファイルハンドラー（デバッグログ用）
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
file_handler = logging.FileHandler(log_dir / "elevenlabs_debug.log", encoding='utf-8')
file_handler.setLevel(logging.DEBUG)

# フォーマッターの設定
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ElevenLabs設定
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

if not ELEVENLABS_API_KEY:
    logger.warning("ELEVENLABS_API_KEY環境変数が設定されていません。")
    ELEVENLABS_API_KEY = "your-api-key"

def transcribe_audio_file(audio_file_path, language_code=None):
    """ElevenLabs Scribeで音声ファイルを文字起こしする
    
    Args:
        audio_file_path: 音声ファイルのパス
        language_code: 言語コード (例: "ja", "en") - Noneの場合は自動検出
    
    Returns:
        文字起こし結果のテキスト または (None, エラーメッセージ) のタプル
    """
    logger.info(f"処理開始: {audio_file_path}")
    logger.debug(f"ファイルパス: {audio_file_path}, 言語コード: {language_code}")
    logger.debug(f"APIキー: {ELEVENLABS_API_KEY[:10]}...（マスク）")
    
    # ファイル情報をログ
    try:
        file_size = os.path.getsize(audio_file_path) / (1024 * 1024)  # MB
        logger.debug(f"ファイルサイズ: {file_size:.2f} MB")
    except Exception as e:
        logger.error(f"ファイル情報取得エラー: {e}")
    
    try:
        # ElevenLabsクライアントの初期化（タイムアウトを30分に設定）
        logger.debug("ElevenLabsクライアントを初期化中...")
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY, timeout=1800.0)
        logger.debug("ElevenLabsクライアントの初期化完了（タイムアウト: 1800秒）")
        
        logger.info(f"文字起こしを実行中: {Path(audio_file_path).name}")
        
        # 音声ファイルを開く
        with open(audio_file_path, "rb") as audio_file:
            logger.debug("音声ファイルを読み込み中...")
            # Speech-to-Text変換を実行
            # Scribe v1モデルを使用
            logger.debug("API呼び出し開始...")
            # APIパラメータを構築
            api_params = {
                "file": audio_file,
                "model_id": "scribe_v1",  # または "scribe_v1_experimental"
                "tag_audio_events": True  # 笑い声、拍手などの非音声イベントもタグ付け
            }
            
            # language_codeが指定されていて、空文字列でない場合のみ追加
            if language_code and language_code.strip():
                api_params["language_code"] = language_code
                logger.debug(f"言語コードを指定: {language_code}")
            else:
                logger.debug("言語コードは自動検出モード")
            
            result = client.speech_to_text.convert(**api_params)
            logger.debug("API呼び出し完了")
        
        # 結果の処理
        if result.text:
            logger.info(f"文字起こし成功: {len(result.text)}文字")
            return result.text
        else:
            # 結果が複数のセグメントに分かれている場合
            if hasattr(result, 'segments'):
                transcription = " ".join([segment.text for segment in result.segments])
                logger.info(f"文字起こし成功（セグメント結合）: {len(transcription)}文字")
                return transcription
            logger.warning("文字起こし結果が空です")
            return None
            
    except Exception as e:
        error_msg = f"{audio_file_path} の処理中にエラーが発生しました: {type(e).__name__}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        # エラー情報を含むタプルを返す
        return (None, error_msg)

def save_transcription(filename, transcription, output_dir):
    """文字起こし結果をテキストファイルとして保存"""
    output_filename = Path(filename).stem + "_transcription.txt"
    output_path = output_dir / output_filename
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"ファイル名: {filename}\n")
        f.write(f"文字起こし日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"サービス: ElevenLabs (Scribe v1)\n")
        f.write("-" * 50 + "\n")
        f.write(transcription)
    
    return output_path

def get_file_size_mb(file_path):
    """ファイルサイズをMBで取得"""
    return os.path.getsize(file_path) / (1024 * 1024)

def process_all_audio_files():
    """dataディレクトリ内のすべての音声ファイルを処理"""
    # パスの設定
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    output_dir = base_dir / "transcriptions" / "elevenlabs"
    
    # 出力ディレクトリの作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # すべての結果を保存するためのデータ
    all_results = {}
    
    # サポートされている音声フォーマット（主要な音声/動画フォーマット）
    audio_extensions = ["*.mp3", "*.mp4", "*.wav", "*.m4a", "*.flac", "*.webm", "*.ogg", "*.aac", "*.mov", "*.avi"]
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(data_dir.glob(ext))
    
    if not audio_files:
        logger.warning("音声ファイルが見つかりません。")
        return
    
    logger.info(f"{len(audio_files)}個の音声ファイルを処理します...\n")
    
    for i, audio_file in enumerate(audio_files, 1):
        logger.info(f"[{i}/{len(audio_files)}] 処理中: {audio_file.name}")
        
        # ファイルサイズチェック（1GB制限）
        file_size_mb = get_file_size_mb(audio_file)
        if file_size_mb > 1024:  # 1GB = 1024MB
            logger.warning(f"  → スキップ: ファイルサイズが1GBを超えています ({file_size_mb:.2f}MB)")
            continue
        
        # 文字起こし実行（日本語を想定）
        result = transcribe_audio_file(audio_file, language_code="ja")
        
        # タプルでエラー情報が返ってきた場合の処理
        if isinstance(result, tuple) and result[0] is None:
            transcription = None
            error_msg = result[1]
            logger.error(f"  → スキップ: {error_msg}")
        else:
            transcription = result
        
        if transcription:
            # 個別のテキストファイルとして保存
            output_path = save_transcription(audio_file.name, transcription, output_dir)
            logger.info(f"  → 保存完了: {output_path}")
            
            # 全体の結果に追加
            all_results[audio_file.name] = {
                "transcription": transcription,
                "timestamp": datetime.now().isoformat(),
                "file_size_mb": round(file_size_mb, 2)
            }
        else:
            logger.error(f"  → スキップ: エラーが発生しました")
    
    # すべての結果をJSONファイルとしても保存
    json_output_path = output_dir / f"all_transcriptions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n処理完了！")
    logger.info(f"個別の文字起こし結果: {output_dir}")
    logger.info(f"全体のJSON結果: {json_output_path}")

if __name__ == "__main__":
    # API認証情報の確認
    if not ELEVENLABS_API_KEY or ELEVENLABS_API_KEY == "your-api-key":
        logger.error("エラー: ElevenLabsの認証情報が設定されていません。")
        logger.error("ELEVENLABS_API_KEY環境変数を設定してください。")
        logger.info("\nAPIキーの取得方法:")
        logger.info("1. https://elevenlabs.io にアクセス")
        logger.info("2. アカウントを作成またはログイン")
        logger.info("3. ダッシュボードでAPIキーを作成")
        exit(1)
    
    process_all_audio_files()