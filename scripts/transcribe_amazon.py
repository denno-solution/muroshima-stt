import os
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
import boto3
from botocore.exceptions import BotoCore3Error, ClientError

# AWS設定
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
S3_BUCKET = os.getenv("AWS_S3_BUCKET", "your-s3-bucket-name")

# AWSクライアントの初期化
s3_client = boto3.client('s3', region_name=AWS_REGION)
transcribe_client = boto3.client('transcribe', region_name=AWS_REGION)

def upload_to_s3(file_path, bucket, key):
    """ファイルをS3にアップロード"""
    try:
        s3_client.upload_file(file_path, bucket, key)
        return f"s3://{bucket}/{key}"
    except Exception as e:
        print(f"S3アップロードエラー: {e}")
        return None

def delete_from_s3(bucket, key):
    """S3からファイルを削除"""
    try:
        s3_client.delete_object(Bucket=bucket, Key=key)
    except Exception as e:
        print(f"S3削除エラー: {e}")

def transcribe_audio_file(audio_file_path, language_code="ja-JP"):
    """Amazon Transcribeで音声ファイルを文字起こしする
    
    Args:
        audio_file_path: 音声ファイルのパス
        language_code: 言語コード (例: "ja-JP", "en-US")
    
    Returns:
        文字起こし結果のテキスト
    """
    try:
        # ユニークなジョブ名を生成
        job_name = f"transcription-{uuid.uuid4()}"
        
        # ファイル名と拡張子を取得
        file_path = Path(audio_file_path)
        file_extension = file_path.suffix.lower().replace('.', '')
        
        # S3キーを生成
        s3_key = f"transcriptions/{job_name}/{file_path.name}"
        
        # S3にアップロード
        print(f"  → S3にアップロード中...")
        s3_uri = upload_to_s3(str(audio_file_path), S3_BUCKET, s3_key)
        if not s3_uri:
            return None
        
        # サポートされているメディアフォーマットをマッピング
        format_mapping = {
            'mp3': 'mp3',
            'mp4': 'mp4',
            'wav': 'wav',
            'flac': 'flac',
            'm4a': 'm4a',
            'webm': 'webm'
        }
        
        media_format = format_mapping.get(file_extension, 'mp3')
        
        # 文字起こしジョブを開始
        print(f"  → 文字起こしジョブを開始...")
        transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': s3_uri},
            MediaFormat=media_format,
            LanguageCode=language_code,
            Settings={
                'ShowSpeakerLabels': False,  # 話者の識別
                'MaxSpeakerLabels': 2  # 最大話者数
            }
        )
        
        # ジョブの完了を待つ
        max_tries = 60
        while max_tries > 0:
            max_tries -= 1
            job = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
            job_status = job['TranscriptionJob']['TranscriptionJobStatus']
            
            if job_status in ['COMPLETED', 'FAILED']:
                if job_status == 'COMPLETED':
                    # 結果を取得
                    transcript_uri = job['TranscriptionJob']['Transcript']['TranscriptFileUri']
                    
                    # 結果をダウンロード
                    import urllib.request
                    with urllib.request.urlopen(transcript_uri) as response:
                        result_json = json.loads(response.read().decode('utf-8'))
                    
                    # 文字起こしテキストを抽出
                    transcription = result_json['results']['transcripts'][0]['transcript']
                    
                    # クリーンアップ
                    transcribe_client.delete_transcription_job(TranscriptionJobName=job_name)
                    delete_from_s3(S3_BUCKET, s3_key)
                    
                    return transcription
                else:
                    print(f"  → ジョブ失敗: {job_name}")
                    delete_from_s3(S3_BUCKET, s3_key)
                    return None
                break
            else:
                time.sleep(5)
        
        # タイムアウト
        print(f"  → タイムアウト: {job_name}")
        delete_from_s3(S3_BUCKET, s3_key)
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
        f.write(f"サービス: Amazon Transcribe\n")
        f.write("-" * 50 + "\n")
        f.write(transcription)
    
    return output_path

def process_all_audio_files():
    """dataディレクトリ内のすべての音声ファイルを処理"""
    # パスの設定
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    output_dir = base_dir / "transcriptions" / "amazon"
    
    # 出力ディレクトリの作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # すべての結果を保存するためのデータ
    all_results = {}
    
    # サポートされている音声フォーマット
    audio_extensions = ["*.mp3", "*.mp4", "*.wav", "*.m4a", "*.flac", "*.webm"]
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(data_dir.glob(ext))
    
    if not audio_files:
        print("音声ファイルが見つかりません。")
        return
    
    print(f"{len(audio_files)}個の音声ファイルを処理します...\n")
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"[{i}/{len(audio_files)}] 処理中: {audio_file.name}")
        
        # 文字起こし実行
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
    # AWS認証情報の確認
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"AWS アカウント: {identity['Account']}")
    except Exception as e:
        print("警告: AWS認証情報が設定されていません。")
        print("AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_BUCKET環境変数を設定してください。")
        exit(1)
    
    process_all_audio_files()