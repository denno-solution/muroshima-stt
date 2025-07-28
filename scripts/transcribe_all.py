#!/usr/bin/env python3
"""
すべてのSTTサービスを実行して結果を比較表に保存するスクリプト

使用方法:
    python transcribe_all.py [--sample]
    
    --sample: 最初の3ファイルのみ処理（テスト用）
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path
from datetime import datetime

# 各STTサービスのモジュールをインポート
sys.path.append(str(Path(__file__).parent / "src"))

def check_requirements():
    """必要なパッケージがインストールされているか確認"""
    required_packages = {
        'openai': 'openai',
        'google-cloud-speech': 'google.cloud.speech_v2',
        'boto3': 'boto3',
        'azure-cognitiveservices-speech': 'azure.cognitiveservices.speech',
        'elevenlabs': 'elevenlabs'
    }
    
    missing_packages = []
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("以下のパッケージがインストールされていません:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n以下のコマンドでインストールしてください:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def run_openai_transcription(audio_files):
    """OpenAI Whisperで文字起こし"""
    try:
        from transcribe_openai import transcribe_audio_file
        results = {}
        
        print("\n=== OpenAI gpt-4o-transcribe ===")
        for i, audio_file in enumerate(audio_files, 1):
            print(f"[{i}/{len(audio_files)}] {audio_file.name}")
            transcription = transcribe_audio_file(audio_file)
            if transcription:
                results[audio_file.name] = transcription
            else:
                results[audio_file.name] = "エラー"
        
        return results
    except Exception as e:
        print(f"OpenAIの処理中にエラー: {e}")
        return {}

def run_google_transcription(audio_files):
    """Google Cloud Speech-to-Textで文字起こし"""
    try:
        from transcribe_google import transcribe_audio_file
        results = {}
        
        print("\n=== Google Cloud Speech-to-Text (Chirp) ===")
        for i, audio_file in enumerate(audio_files, 1):
            print(f"[{i}/{len(audio_files)}] {audio_file.name}")
            transcription = transcribe_audio_file(audio_file, model="chirp")
            if transcription:
                results[audio_file.name] = transcription
            else:
                results[audio_file.name] = "エラー"
        
        return results
    except Exception as e:
        print(f"Google Cloudの処理中にエラー: {e}")
        return {}

def run_amazon_transcription(audio_files):
    """Amazon Transcribeで文字起こし"""
    try:
        from transcribe_amazon import transcribe_audio_file
        results = {}
        
        print("\n=== Amazon Transcribe ===")
        for i, audio_file in enumerate(audio_files, 1):
            print(f"[{i}/{len(audio_files)}] {audio_file.name}")
            transcription = transcribe_audio_file(audio_file)
            if transcription:
                results[audio_file.name] = transcription
            else:
                results[audio_file.name] = "エラー"
        
        return results
    except Exception as e:
        print(f"Amazon Transcribeの処理中にエラー: {e}")
        return {}

def run_azure_transcription(audio_files):
    """Microsoft Azure AI Speechで文字起こし"""
    try:
        from transcribe_azure import transcribe_audio_file_simple
        results = {}
        
        print("\n=== Microsoft Azure AI Speech ===")
        for i, audio_file in enumerate(audio_files, 1):
            print(f"[{i}/{len(audio_files)}] {audio_file.name}")
            transcription = transcribe_audio_file_simple(audio_file)
            if transcription:
                results[audio_file.name] = transcription
            else:
                results[audio_file.name] = "エラー"
        
        return results
    except Exception as e:
        print(f"Azure AI Speechの処理中にエラー: {e}")
        return {}

def run_elevenlabs_transcription(audio_files):
    """ElevenLabsで文字起こし"""
    try:
        from transcribe_elevenlabs import transcribe_audio_file
        results = {}
        
        print("\n=== ElevenLabs (Scribe v1) ===")
        for i, audio_file in enumerate(audio_files, 1):
            print(f"[{i}/{len(audio_files)}] {audio_file.name}")
            transcription = transcribe_audio_file(audio_file, language_code="ja")
            if transcription:
                results[audio_file.name] = transcription
            else:
                results[audio_file.name] = "エラー"
        
        return results
    except Exception as e:
        print(f"ElevenLabsの処理中にエラー: {e}")
        return {}

def save_comparison_csv(all_results, output_path):
    """比較結果をCSVファイルに保存"""
    # ヘッダー
    headers = ['音源ファイル', 'OpenAI', 'Google Cloud', 'Amazon Transcribe', 'Azure AI Speech', 'ElevenLabs']
    
    # すべてのファイル名を取得
    all_files = set()
    for service_results in all_results.values():
        all_files.update(service_results.keys())
    
    # CSVに書き込み
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for filename in sorted(all_files):
            row = [filename.replace('.mp3', '')]  # 拡張子を除去
            
            # 各サービスの結果を追加
            for service in ['openai', 'google', 'amazon', 'azure', 'elevenlabs']:
                if service in all_results and filename in all_results[service]:
                    row.append(all_results[service][filename])
                else:
                    row.append('')
            
            writer.writerow(row)
    
    print(f"\n比較結果を保存しました: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='すべてのSTTサービスで文字起こしを実行')
    parser.add_argument('--sample', action='store_true', help='最初の3ファイルのみ処理')
    args = parser.parse_args()
    
    # 必要なパッケージの確認
    if not check_requirements():
        return
    
    # パスの設定
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    output_dir = base_dir / "transcriptions"
    
    # 音声ファイルを取得
    audio_files = list(data_dir.glob("*.mp3"))
    if not audio_files:
        print("音声ファイルが見つかりません。")
        return
    
    # サンプルモードの場合は最初の3ファイルのみ
    if args.sample:
        audio_files = audio_files[:3]
        print(f"サンプルモード: 最初の{len(audio_files)}ファイルのみ処理します")
    
    print(f"\n{len(audio_files)}個の音声ファイルを処理します...")
    
    # 結果を格納する辞書
    all_results = {}
    
    # 各サービスで文字起こしを実行
    services = [
        ('openai', run_openai_transcription),
        ('google', run_google_transcription),
        ('amazon', run_amazon_transcription),
        ('azure', run_azure_transcription),
        ('elevenlabs', run_elevenlabs_transcription)
    ]
    
    for service_name, service_func in services:
        try:
            results = service_func(audio_files)
            if results:
                all_results[service_name] = results
        except Exception as e:
            print(f"{service_name}の実行中にエラー: {e}")
    
    # 比較結果をCSVに保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = output_dir / f"STT_比較結果_{timestamp}.csv"
    save_comparison_csv(all_results, csv_path)
    
    # JSON形式でも保存
    json_path = output_dir / f"STT_比較結果_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"JSON結果も保存しました: {json_path}")
    print("\n処理完了！")

if __name__ == "__main__":
    main()