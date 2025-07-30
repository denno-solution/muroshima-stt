import json
import os
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI

# APIキーを環境変数から取得（より安全）
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    api_key = "aaa"
    print("警告: APIキーがハードコードされています。環境変数OPENAI_API_KEYの使用を推奨します。")

client = OpenAI(api_key=api_key)

def analyze_transcription_quality(text):
    """文字起こし結果の品質を分析"""
    words = text.split("、")
    total_words = len(words)

    if total_words == 0:
        return {"quality": "empty", "repetition_rate": 0}

    # 単語の頻度を計算
    word_freq = {}
    for word in words:
        word = word.strip()
        if word:
            word_freq[word] = word_freq.get(word, 0) + 1

    # 最も頻出する単語とその頻度
    if word_freq:
        most_common_word = max(word_freq, key=word_freq.get)
        max_freq = word_freq[most_common_word]
        repetition_rate = max_freq / total_words

        return {
            "quality": "repetitive" if repetition_rate > 0.5 else "normal",
            "repetition_rate": repetition_rate,
            "most_common_word": most_common_word,
            "frequency": max_freq,
            "total_words": total_words,
            "unique_words": len(word_freq)
        }

    return {"quality": "unknown", "repetition_rate": 0}

def transcribe_with_multiple_models(audio_file_path):
    """複数のモデルで文字起こしを試みる"""
    results = {}

    # 1. gpt-4o-transcribeを試す
    print("  1. gpt-4o-transcribeで試行中...")
    try:
        with open(audio_file_path, "rb") as audio_file:
            start_time = time.time()
            transcription = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=audio_file,
                language="ja"
            )
            elapsed_time = time.time() - start_time

            results["gpt-4o-transcribe"] = {
                "text": transcription.text,
                "time": elapsed_time,
                "analysis": analyze_transcription_quality(transcription.text)
            }
            print(f"    完了 (処理時間: {elapsed_time:.2f}秒)")
    except Exception as e:
        results["gpt-4o-transcribe"] = {"error": str(e)}
        print(f"    エラー: {e}")

    # 2. gpt-4o-transcribeにプロンプト付きで再試行
    print("  2. gpt-4o-transcribe（プロンプト付き）で試行中...")
    try:
        with open(audio_file_path, "rb") as audio_file:
            start_time = time.time()
            transcription = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=audio_file,
                language="ja",
                prompt="工場での機械設定変更の音声。数値、温度、時間、速度などのパラメータを正確に記録。"
            )
            elapsed_time = time.time() - start_time

            results["gpt-4o-transcribe-prompted"] = {
                "text": transcription.text,
                "time": elapsed_time,
                "analysis": analyze_transcription_quality(transcription.text)
            }
            print(f"    完了 (処理時間: {elapsed_time:.2f}秒)")
    except Exception as e:
        results["gpt-4o-transcribe-prompted"] = {"error": str(e)}
        print(f"    エラー: {e}")

    # 3. whisper-1を試す
    print("  3. whisper-1で試行中...")
    try:
        with open(audio_file_path, "rb") as audio_file:
            start_time = time.time()
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="ja"
            )
            elapsed_time = time.time() - start_time

            results["whisper-1"] = {
                "text": transcription.text,
                "time": elapsed_time,
                "analysis": analyze_transcription_quality(transcription.text)
            }
            print(f"    完了 (処理時間: {elapsed_time:.2f}秒)")
    except Exception as e:
        results["whisper-1"] = {"error": str(e)}
        print(f"    エラー: {e}")

    return results

def save_debug_results(filename, results, output_dir):
    """デバッグ結果を保存"""
    output_filename = Path(filename).stem + "_debug.json"
    output_path = output_dir / output_filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "filename": filename,
            "timestamp": datetime.now().isoformat(),
            "results": results
        }, f, ensure_ascii=False, indent=2)

    return output_path

def test_problematic_files():
    """問題のあるファイルをテスト"""
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    output_dir = base_dir / "transcriptions/openai_debug"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 問題のあるファイルをテスト
    problematic_files = ["レコーディング (8).mp3", "レコーディング (17).mp3"]

    for filename in problematic_files:
        audio_file = data_dir / filename
        if audio_file.exists():
            print(f"\n処理中: {filename}")
            print("-" * 50)

            results = transcribe_with_multiple_models(audio_file)

            # 結果を保存
            output_path = save_debug_results(filename, results, output_dir)
            print(f"\nデバッグ結果を保存: {output_path}")

            # 結果のサマリーを表示
            print("\n結果サマリー:")
            for model, result in results.items():
                if "error" in result:
                    print(f"  {model}: エラー - {result['error']}")
                else:
                    analysis = result["analysis"]
                    print(f"  {model}:")
                    print(f"    - 品質: {analysis['quality']}")
                    print(f"    - 繰り返し率: {analysis.get('repetition_rate', 0):.2%}")
                    if "most_common_word" in analysis:
                        print(f"    - 最頻出語: '{analysis['most_common_word']}' ({analysis['frequency']}回/{analysis['total_words']}語)")
                    print(f"    - 処理時間: {result['time']:.2f}秒")
        else:
            print(f"ファイルが見つかりません: {filename}")

if __name__ == "__main__":
    test_problematic_files()