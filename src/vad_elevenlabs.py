"""
VAD対応のElevenLabs STT処理モジュール
webrtcvadで無音部分を除去してからElevenLabs APIに送信することでコスト削減
"""

import os
import requests
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from dotenv import load_dotenv

from vad_processor import VADProcessor

# .envファイルを読み込む
load_dotenv()

logger = logging.getLogger(__name__)

# ElevenLabs Speech-to-Text API設定
ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/speech-to-text"
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")


class VADElevenLabsSTT:
    """VAD対応のElevenLabs STT処理クラス"""
    
    def __init__(self, api_key: Optional[str] = None, vad_aggressiveness: int = 2):
        """
        Args:
            api_key: ElevenLabs APIキー（Noneの場合は環境変数から取得）
            vad_aggressiveness: VADの厳しさ (0=ゆるい, 3=厳しい)
        """
        self.api_key = api_key or ELEVENLABS_API_KEY
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY が設定されていません")
        
        self.vad_processor = VADProcessor(aggressiveness=vad_aggressiveness)
        self.session = requests.Session()
        self.session.headers.update({"xi-api-key": self.api_key})
    
    def transcribe_with_vad(
        self,
        audio_bytes: bytes,
        min_speech_ms: int = 300,
        merge_gap_ms: int = 300,
        language_code: Optional[str] = None,
        model_id: str = "scribe_v1",
        diarize: bool = True,
        tag_audio_events: bool = False,
        use_multi_channel: bool = False
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        VAD処理を適用してからElevenLabs STTで文字起こし
        
        Args:
            audio_bytes: 入力音声データ
            min_speech_ms: 最小スピーチ長（ms）
            merge_gap_ms: 区間マージ許容ギャップ（ms）
            language_code: 言語コード（自動検出の場合はNone）
            model_id: ElevenLabsモデルID
            diarize: 話者分離を有効にするか
            tag_audio_events: 音声イベントタグ付けを有効にするか
            use_multi_channel: マルチチャンネル処理を有効にするか
            
        Returns:
            Tuple[Optional[str], Dict[str, Any]]: (文字起こし結果, 統計情報・メタデータ)
        """
        try:
            logger.info("VAD処理を開始...")
            
            # Step 1: VAD処理で無音部分を除去
            processed_audio, vad_stats = self.vad_processor.process_audio(
                audio_bytes,
                min_speech_ms=min_speech_ms,
                merge_gap_ms=merge_gap_ms
            )
            
            logger.info(f"VAD処理完了。圧縮率: {vad_stats['compression_ratio']:.2%}")
            
            # Step 2: ElevenLabs APIに送信
            logger.info("ElevenLabs APIに送信中...")
            
            # APIパラメータを構築
            data = {
                "model_id": model_id,
                "file_format": "pcm_s16le_16",  # 16kHz/mono/16bit PCM
                "diarize": str(diarize).lower(),
                "tag_audio_events": str(tag_audio_events).lower(),
                "use_multi_channel": str(use_multi_channel).lower()
            }
            
            # 言語コードが指定されている場合は追加
            if language_code:
                data["language_code"] = language_code
            
            # ファイルデータ
            files = {
                "file": ("audio.pcm", processed_audio, "application/octet-stream")
            }
            
            # API呼び出し
            response = self.session.post(
                ELEVENLABS_API_URL,
                data=data,
                files=files,
                timeout=120
            )
            
            response.raise_for_status()
            result = response.json()
            
            logger.info("ElevenLabs API処理完了")
            
            # 結果の処理
            transcript = result.get("text", "")
            
            # メタデータと統計情報をマージ
            metadata = {
                "vad_stats": vad_stats,
                "elevenlabs_response": result,
                "api_parameters": {
                    "model_id": model_id,
                    "language_code": language_code,
                    "diarize": diarize,
                    "tag_audio_events": tag_audio_events,
                    "use_multi_channel": use_multi_channel
                },
                "processing_info": {
                    "original_duration_seconds": vad_stats["original_duration_ms"] / 1000,
                    "processed_duration_seconds": vad_stats["speech_duration_ms"] / 1000,
                    "cost_reduction_ratio": 1 - vad_stats["compression_ratio"]
                }
            }
            
            return transcript, metadata
            
        except requests.exceptions.RequestException as e:
            logger.error(f"ElevenLabs API エラー: {e}")
            return None, {"error": f"API Error: {e}", "vad_stats": vad_stats if 'vad_stats' in locals() else {}}
        
        except Exception as e:
            logger.error(f"VAD処理エラー: {e}")
            return None, {"error": f"Processing Error: {e}"}
    
    def transcribe_file_with_vad(
        self,
        file_path: str,
        **kwargs
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        ファイルからVAD処理付きで文字起こし
        
        Args:
            file_path: 音声ファイルのパス
            **kwargs: transcribe_with_vad()のパラメータ
            
        Returns:
            Tuple[Optional[str], Dict[str, Any]]: (文字起こし結果, 統計情報・メタデータ)
        """
        try:
            with open(file_path, 'rb') as f:
                audio_bytes = f.read()
            
            return self.transcribe_with_vad(audio_bytes, **kwargs)
            
        except FileNotFoundError:
            logger.error(f"ファイルが見つかりません: {file_path}")
            return None, {"error": f"File not found: {file_path}"}
        
        except Exception as e:
            logger.error(f"ファイル読み込みエラー: {e}")
            return None, {"error": f"File read error: {e}"}


def transcribe_audio_file_with_vad(
    audio_file_path: str,
    language_code: Optional[str] = None,
    vad_aggressiveness: int = 2,
    min_speech_ms: int = 300,
    merge_gap_ms: int = 300,
    **elevenlabs_params
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    VAD対応の文字起こし関数（既存のスクリプトとの互換性のため）
    
    Args:
        audio_file_path: 音声ファイルのパス
        language_code: 言語コード
        vad_aggressiveness: VADの厳しさ
        min_speech_ms: 最小スピーチ長（ms）
        merge_gap_ms: 区間マージ許容ギャップ（ms）
        **elevenlabs_params: ElevenLabs APIの追加パラメータ
        
    Returns:
        Tuple[Optional[str], Dict[str, Any]]: (文字起こし結果, 統計情報・メタデータ)
    """
    try:
        stt = VADElevenLabsSTT(vad_aggressiveness=vad_aggressiveness)
        return stt.transcribe_file_with_vad(
            audio_file_path,
            language_code=language_code,
            min_speech_ms=min_speech_ms,
            merge_gap_ms=merge_gap_ms,
            **elevenlabs_params
        )
    except Exception as e:
        logger.error(f"VAD対応文字起こし処理エラー: {e}")
        return None, {"error": str(e)}


# 使用例
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) != 2:
        print("Usage: python vad_elevenlabs.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    try:
        transcript, metadata = transcribe_audio_file_with_vad(
            audio_file,
            language_code="ja",  # 日本語指定
            vad_aggressiveness=2,
            min_speech_ms=300,
            merge_gap_ms=300
        )
        
        if transcript:
            print(f"Transcript: {transcript}")
            print(f"Original duration: {metadata['vad_stats']['original_duration_ms']} ms")
            print(f"Speech duration: {metadata['vad_stats']['speech_duration_ms']} ms")
            print(f"Compression ratio: {metadata['vad_stats']['compression_ratio']:.2%}")
            print(f"Cost reduction: {metadata['processing_info']['cost_reduction_ratio']:.2%}")
        else:
            print(f"Error: {metadata.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)