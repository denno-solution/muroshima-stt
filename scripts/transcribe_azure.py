import os
import json
from datetime import datetime
from pathlib import Path
import azure.cognitiveservices.speech as speechsdk

# Azure設定
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SERVICE_REGION = os.getenv("AZURE_SERVICE_REGION", "westus")

if not AZURE_SPEECH_KEY:
    print("警告: AZURE_SPEECH_KEY環境変数が設定されていません。")
    AZURE_SPEECH_KEY = "your-speech-key"

def transcribe_audio_file(audio_file_path, language="ja-JP"):
    """Microsoft Azure AI Speechで音声ファイルを文字起こしする
    
    Args:
        audio_file_path: 音声ファイルのパス
        language: 言語コード (例: "ja-JP", "en-US")
    
    Returns:
        文字起こし結果のテキスト
    """
    try:
        # Speech設定の作成
        speech_config = speechsdk.SpeechConfig(
            subscription=AZURE_SPEECH_KEY,
            region=AZURE_SERVICE_REGION
        )
        speech_config.speech_recognition_language = language
        
        # 音声ファイルの設定
        audio_input = speechsdk.AudioConfig(filename=str(audio_file_path))
        
        # 音声認識器の作成
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_input
        )
        
        # 結果を格納する変数
        all_results = []
        done = False
        
        def stop_cb(evt):
            """認識が停止したときのコールバック"""
            nonlocal done
            done = True
        
        def recognized_cb(evt):
            """音声が認識されたときのコールバック"""
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                all_results.append(evt.result.text)
        
        # イベントハンドラーを接続
        speech_recognizer.recognized.connect(recognized_cb)
        speech_recognizer.session_stopped.connect(stop_cb)
        speech_recognizer.canceled.connect(stop_cb)
        
        # 連続認識を開始
        print(f"  → 音声認識を開始...")
        speech_recognizer.start_continuous_recognition()
        
        # 認識が完了するまで待機
        import time
        while not done:
            time.sleep(0.5)
        
        speech_recognizer.stop_continuous_recognition()
        
        # 結果を結合
        transcription = " ".join(all_results)
        return transcription if transcription else None
        
    except Exception as e:
        print(f"エラー: {audio_file_path} の処理中にエラーが発生しました: {e}")
        return None

def transcribe_audio_file_simple(audio_file_path, language="ja-JP"):
    """シンプルな単発認識（短い音声ファイル用）"""
    try:
        # Speech設定の作成
        speech_config = speechsdk.SpeechConfig(
            subscription=AZURE_SPEECH_KEY,
            region=AZURE_SERVICE_REGION
        )
        speech_config.speech_recognition_language = language
        
        # 音声ファイルの設定
        audio_input = speechsdk.AudioConfig(filename=str(audio_file_path))
        
        # 音声認識器の作成
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_input
        )
        
        print(f"  → 音声認識を実行中...")
        
        # 単発認識を実行
        result = speech_recognizer.recognize_once()
        
        # 結果の処理
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text
        elif result.reason == speechsdk.ResultReason.NoMatch:
            print(f"  → 音声が認識できませんでした: {result.no_match_details}")
            return None
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print(f"  → 認識がキャンセルされました: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print(f"  → エラー詳細: {cancellation_details.error_details}")
            return None
            
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
        f.write(f"サービス: Microsoft Azure AI Speech\n")
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
    output_dir = base_dir / "transcriptions" / "azure"
    
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
    
    print(f"{len(audio_files)}個の音声ファイルを処理します...\n")
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"[{i}/{len(audio_files)}] 処理中: {audio_file.name}")
        
        # ファイルサイズをチェック（小さいファイルは単発認識を使用）
        file_size_mb = get_file_size_mb(audio_file)
        
        # 文字起こし実行
        if file_size_mb < 5:  # 5MB未満は単発認識
            transcription = transcribe_audio_file_simple(audio_file)
        else:  # それ以上は連続認識
            transcription = transcribe_audio_file(audio_file)
        
        if transcription:
            # 個別のテキストファイルとして保存
            output_path = save_transcription(audio_file.name, transcription, output_dir)
            print(f"  → 保存完了: {output_path}")
            
            # 全体の結果に追加
            all_results[audio_file.name] = {
                "transcription": transcription,
                "timestamp": datetime.now().isoformat()
            }
        else:
            print(f"  → スキップ: エラーが発生しました")
    
    # すべての結果をJSONファイルとしても保存
    json_output_path = output_dir / f"all_transcriptions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n処理完了！")
    print(f"個別の文字起こし結果: {output_dir}")
    print(f"全体のJSON結果: {json_output_path}")

if __name__ == "__main__":
    # Azure認証情報の確認
    if not AZURE_SPEECH_KEY or AZURE_SPEECH_KEY == "your-speech-key":
        print("エラー: Azure Speech Serviceの認証情報が設定されていません。")
        print("AZURE_SPEECH_KEY環境変数を設定してください。")
        print("また、AZURE_SERVICE_REGION環境変数でリージョンを指定できます（デフォルト: westus）")
        exit(1)
    
    process_all_audio_files()