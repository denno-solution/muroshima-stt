import json
import os
from datetime import datetime
from pathlib import Path

from openai import OpenAI

# APIキーを環境変数から取得（より安全）
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    # 環境変数が設定されていない場合は、ここにAPIキーを設定
    api_key = "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"
    print("警告: APIキーがハードコードされています。環境変数OPENAI_API_KEYの使用を推奨します。")

client = OpenAI(api_key=api_key)

def transcribe_audio_file(audio_file_path):
    """単一の音声ファイルを文字起こしする（改善版）"""
    try:
        with open(audio_file_path, "rb") as audio_file:
            # 1. gpt-4o-transcribeで試す
            try:
                transcription = client.audio.transcriptions.create(
                    model="gpt-4o-transcribe",
                    file=audio_file,
                    language="ja",  # 日本語を明示的に指定
                    prompt="これは工場での機械設定に関する音声です。数値、温度、時間、速度などのパラメータ変更を正確に記録してください。"  # プロンプトを追加
                )
                
                # 結果をチェック（繰り返しパターンの検出）
                text = transcription.text
                words = text.split("、")
                if len(words) > 10:
                    # 最初の10語をチェック
                    first_word = words[0]
                    repetition_count = sum(1 for word in words[:10] if word == first_word)
                    
                    # 80%以上が同じ単語の場合は問題ありと判断
                    if repetition_count >= 8:
                        print(f"  警告: 繰り返しパターンを検出しました。whisper-1にフォールバック...")
                        raise Exception("Repetitive pattern detected")
                
                return text
                
            except Exception as e:
                print(f"  gpt-4o-transcribeでエラー: {e}")
                # 2. whisper-1にフォールバック
                audio_file.seek(0)  # ファイルポインタをリセット
                print("  whisper-1モデルで再試行中...")
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="ja"
                )
                return transcription.text
                
    except Exception as e:
        print(f"エラー: {audio_file_path} の処理中にエラーが発生しました: {e}")
        return None

def save_transcription(filename, transcription, output_dir, model_used="unknown"):
    """文字起こし結果をテキストファイルとして保存"""
    output_filename = Path(filename).stem + "_transcription.txt"
    output_path = output_dir / output_filename

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"ファイル名: {filename}\n")
        f.write(f"文字起こし日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"使用モデル: {model_used}\n")
        f.write("-" * 50 + "\n")
        f.write(transcription)

    return output_path

def process_all_audio_files():
    """dataディレクトリ内のすべての音声ファイルを処理"""
    # パスの設定
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    output_dir = base_dir / "transcriptions/openai_improved"

    # 出力ディレクトリの作成
    output_dir.mkdir(parents=True, exist_ok=True)

    # すべての結果を保存するためのデータ
    all_results = {}

    # データディレクトリ内のmp3ファイルを処理
    audio_files = list(data_dir.glob("*.mp3"))

    if not audio_files:
        print("音声ファイルが見つかりません。")
        return

    print(f"{len(audio_files)}個の音声ファイルを処理します...\n")

    for i, audio_file in enumerate(audio_files, 1):
        print(f"[{i}/{len(audio_files)}] 処理中: {audio_file.name}")

        # 文字起こし実行
        transcription = transcribe_audio_file(audio_file)

        if transcription:
            # どのモデルが使用されたかを判定
            model_used = "gpt-4o-transcribe or whisper-1"
            
            # 個別のテキストファイルとして保存
            output_path = save_transcription(audio_file.name, transcription, output_dir, model_used)
            print(f"  → 保存完了: {output_path}")

            # 全体の結果に追加
            all_results[audio_file.name] = {
                "transcription": transcription,
                "timestamp": datetime.now().isoformat(),
                "model": model_used
            }
        else:
            print(f"  → スキップ: {audio_file.name} エラーが発生しました")

    # すべての結果をJSONファイルとしても保存
    json_output_path = output_dir / f"all_transcriptions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\n処理完了！")
    print(f"個別の文字起こし結果: {output_dir}")
    print(f"全体のJSON結果: {json_output_path}")

if __name__ == "__main__":
    process_all_audio_files()