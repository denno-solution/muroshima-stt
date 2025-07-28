import os
import json
from datetime import datetime
from pathlib import Path
from google.api_core.client_options import ClientOptions
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

# Google Cloud プロジェクトID
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
if not PROJECT_ID:
    print("警告: GOOGLE_CLOUD_PROJECT環境変数が設定されていません。")
    PROJECT_ID = "your-project-id"  # ここにプロジェクトIDを設定

def transcribe_audio_file(audio_file_path, model="chirp", language_code="ja-JP"):
    """Google Cloud Speech-to-Text (Chirp/Chirp2)で音声ファイルを文字起こしする
    
    Args:
        audio_file_path: 音声ファイルのパス
        model: 使用するモデル ("chirp" または "chirp_2")
        language_code: 言語コード (例: "ja-JP", "en-US")
    
    Returns:
        文字起こし結果のテキスト
    """
    try:
        # クライアントの初期化
        client = SpeechClient(
            client_options=ClientOptions(
                api_endpoint="us-central1-speech.googleapis.com",
            )
        )
        
        # 音声ファイルを読み込む
        with open(audio_file_path, "rb") as audio_file:
            audio_content = audio_file.read()
        
        # 認識設定
        config = cloud_speech.RecognitionConfig(
            auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
            language_codes=[language_code],
            model=model,  # "chirp" または "chirp_2"
        )
        
        # リクエストの作成
        request = cloud_speech.RecognizeRequest(
            recognizer=f"projects/{PROJECT_ID}/locations/us-central1/recognizers/_",
            config=config,
            content=audio_content,
        )
        
        # 音声認識の実行
        response = client.recognize(request=request)
        
        # 結果を結合
        transcription = ""
        for result in response.results:
            transcription += result.alternatives[0].transcript + " "
        
        return transcription.strip()
        
    except Exception as e:
        print(f"エラー: {audio_file_path} の処理中にエラーが発生しました: {e}")
        return None

def save_transcription(filename, transcription, output_dir):
    """文字起こし結果をテキストファイルとして保存"""
    output_filename = Path(filename).stem + "_transcription.txt"
    output_path = output_dir / output_filename
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"ファイル名: {filename}\n")
        f.write(f"文字起こし日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"サービス: Google Cloud Speech-to-Text (Chirp)\n")
        f.write("-" * 50 + "\n")
        f.write(transcription)
    
    return output_path

def process_all_audio_files(model="chirp"):
    """dataディレクトリ内のすべての音声ファイルを処理"""
    # パスの設定
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    output_dir = base_dir / "transcriptions" / "google"
    
    # 出力ディレクトリの作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # すべての結果を保存するためのデータ
    all_results = {}
    
    # サポートされている音声フォーマット
    audio_extensions = ["*.mp3", "*.mp4", "*.wav", "*.m4a", "*.flac"]
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(data_dir.glob(ext))
    
    if not audio_files:
        print("音声ファイルが見つかりません。")
        return
    
    print(f"{len(audio_files)}個の音声ファイルを処理します (モデル: {model})...\n")
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"[{i}/{len(audio_files)}] 処理中: {audio_file.name}")
        
        # 文字起こし実行
        transcription = transcribe_audio_file(audio_file, model=model)
        
        if transcription:
            # 個別のテキストファイルとして保存
            output_path = save_transcription(audio_file.name, transcription, output_dir)
            print(f"  → 保存完了: {output_path}")
            
            # 全体の結果に追加
            all_results[audio_file.name] = {
                "transcription": transcription,
                "timestamp": datetime.now().isoformat(),
                "model": model
            }
        else:
            print(f"  → スキップ: エラーが発生しました")
    
    # すべての結果をJSONファイルとしても保存
    json_output_path = output_dir / f"all_transcriptions_{model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n処理完了！")
    print(f"個別の文字起こし結果: {output_dir}")
    print(f"全体のJSON結果: {json_output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Google Cloud Speech-to-Text (Chirp/Chirp2)')
    parser.add_argument('--model', choices=['chirp', 'chirp_2'], default='chirp',
                        help='使用するモデル (default: chirp)')
    args = parser.parse_args()
    
    process_all_audio_files(model=args.model)