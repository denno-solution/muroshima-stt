"""
VAD (Voice Activity Detection) 処理モジュール
webrtcvad + FFmpegを使用して無音部分を除去し、ElevenLabsなどのSTT APIのコスト削減を実現
"""

import io
import subprocess
import math
import numpy as np
import webrtcvad
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class VADProcessor:
    """Voice Activity Detection処理クラス"""
    
    def __init__(self, aggressiveness: int = 2):
        """
        Args:
            aggressiveness: VAD の厳しさ (0=ゆるい, 3=厳しい)
        """
        self.aggressiveness = aggressiveness
        self.vad = webrtcvad.Vad(aggressiveness)
    
    def to_pcm16_16k(self, audio_bytes: bytes) -> bytes:
        """
        FFmpegで16kHz/mono/16bit PCMに正規化（pipe入出力）
        
        Args:
            audio_bytes: 入力音声データ（任意のフォーマット）
            
        Returns:
            bytes: 16kHz/mono/16bit PCM データ
        """
        cmd = [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-i", "pipe:0",
            "-ac", "1", "-ar", "16000", "-f", "s16le", "-acodec", "pcm_s16le",
            "pipe:1"
        ]
        try:
            result = subprocess.run(
                cmd, 
                input=audio_bytes, 
                capture_output=True, 
                check=True,
                timeout=120
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg conversion failed: {e.stderr.decode()}")
            raise
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg conversion timed out")
            raise
    
    def detect_speech_segments(
        self, 
        pcm_data: bytes, 
        sample_rate: int = 16000,
        frame_ms: int = 30,
        min_speech_ms: int = 300,
        merge_gap_ms: int = 300
    ) -> List[Tuple[int, int]]:
        """
        PCM音声データから有声区間を検出
        
        Args:
            pcm_data: 16bit PCM音声データ
            sample_rate: サンプリングレート（Hz）
            frame_ms: フレーム長（ms）
            min_speech_ms: 最小スピーチ長（ms）
            merge_gap_ms: 区間マージ許容ギャップ（ms）
            
        Returns:
            List[Tuple[int, int]]: 有声区間のリスト [(start_ms, end_ms), ...]
        """
        frame_bytes = int(sample_rate * (frame_ms / 1000.0)) * 2  # int16 = 2bytes
        frames = []
        
        # フレームに分割
        for i in range(0, len(pcm_data), frame_bytes):
            frame = pcm_data[i:i + frame_bytes]
            frames.append(frame)
        
        # 各フレームで音声検出
        voiced = []
        for frame in frames:
            if len(frame) == frame_bytes:
                try:
                    is_speech = self.vad.is_speech(frame, sample_rate)
                    voiced.append(is_speech)
                except Exception as e:
                    logger.warning(f"VAD processing failed for frame: {e}")
                    voiced.append(False)
            else:
                voiced.append(False)
        
        # 連続するTrue区間を検出
        segments = []
        i = 0
        while i < len(voiced):
            if voiced[i]:
                j = i
                while j < len(voiced) and voiced[j]:
                    j += 1
                
                start_ms = i * frame_ms
                end_ms = j * frame_ms
                
                # 最小スピーチ長をチェック
                if end_ms - start_ms >= min_speech_ms:
                    segments.append((start_ms, end_ms))
                
                i = j
            else:
                i += 1
        
        # 近接区間のマージ
        merged_segments = []
        for segment in segments:
            if (not merged_segments or 
                segment[0] - merged_segments[-1][1] > merge_gap_ms):
                merged_segments.append(list(segment))
            else:
                merged_segments[-1][1] = segment[1]
        
        return [(s, e) for s, e in merged_segments]
    
    def extract_speech_audio(
        self, 
        pcm_data: bytes, 
        segments: List[Tuple[int, int]], 
        sample_rate: int = 16000
    ) -> Tuple[bytes, List[Tuple[int, int, int]]]:
        """
        有声区間を連結し、タイムスタンプマッピングも返す
        
        Args:
            pcm_data: 16bit PCM音声データ
            segments: 有声区間のリスト [(start_ms, end_ms), ...]
            sample_rate: サンプリングレート（Hz）
            
        Returns:
            Tuple[bytes, List[Tuple[int, int, int]]]: 
                (連結音声データ, [(orig_start_ms, orig_end_ms, new_start_ms), ...])
        """
        bytes_per_sample = 2  # int16
        
        def ms_to_byte_index(ms: int) -> int:
            return int(sample_rate * ms / 1000.0) * bytes_per_sample
        
        output_audio = bytearray()
        mapping = []
        new_position_ms = 0
        
        for start_ms, end_ms in segments:
            start_idx = ms_to_byte_index(start_ms)
            end_idx = ms_to_byte_index(end_ms)
            
            # 音声データを抽出
            chunk = pcm_data[start_idx:end_idx]
            output_audio.extend(chunk)
            
            # マッピング情報を保存
            mapping.append((start_ms, end_ms, new_position_ms))
            new_position_ms += (end_ms - start_ms)
        
        return bytes(output_audio), mapping
    
    def process_audio(
        self, 
        audio_bytes: bytes,
        min_speech_ms: int = 300,
        merge_gap_ms: int = 300
    ) -> Tuple[bytes, dict]:
        """
        音声データ全体の処理パイプライン
        
        Args:
            audio_bytes: 入力音声データ（任意のフォーマット）
            min_speech_ms: 最小スピーチ長（ms）
            merge_gap_ms: 区間マージ許容ギャップ（ms）
            
        Returns:
            Tuple[bytes, dict]: (処理済み音声データ, 統計情報)
        """
        logger.info("Starting VAD processing")
        
        # Step 1: PCM変換
        pcm_data = self.to_pcm16_16k(audio_bytes)
        original_duration_ms = len(pcm_data) // 2 // 16  # 16kHz, 16bit
        
        logger.info(f"Original audio duration: {original_duration_ms} ms")
        
        # Step 2: 有声区間検出
        segments = self.detect_speech_segments(
            pcm_data, 
            min_speech_ms=min_speech_ms,
            merge_gap_ms=merge_gap_ms
        )
        
        speech_duration_ms = sum(end - start for start, end in segments)
        logger.info(f"Detected {len(segments)} speech segments, total: {speech_duration_ms} ms")
        
        # Step 3: 有声区間抽出
        processed_audio, mapping = self.extract_speech_audio(pcm_data, segments)
        
        # 統計情報
        stats = {
            "original_duration_ms": original_duration_ms,
            "speech_duration_ms": speech_duration_ms,
            "silence_removed_ms": original_duration_ms - speech_duration_ms,
            "compression_ratio": speech_duration_ms / original_duration_ms if original_duration_ms > 0 else 0,
            "segments_count": len(segments),
            "segments": segments,
            "timestamp_mapping": mapping
        }
        
        logger.info(f"VAD processing complete. Compression ratio: {stats['compression_ratio']:.2%}")
        
        return processed_audio, stats


def create_vad_processor(aggressiveness: int = 2) -> VADProcessor:
    """VADProcessor のファクトリ関数"""
    return VADProcessor(aggressiveness=aggressiveness)


# 使用例とテスト用関数
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) != 2:
        print("Usage: python vad_processor.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    try:
        with open(audio_file, 'rb') as f:
            audio_data = f.read()
        
        processor = VADProcessor(aggressiveness=2)
        processed_audio, stats = processor.process_audio(audio_data)
        
        print(f"Original duration: {stats['original_duration_ms']} ms")
        print(f"Speech duration: {stats['speech_duration_ms']} ms")
        print(f"Silence removed: {stats['silence_removed_ms']} ms")
        print(f"Compression ratio: {stats['compression_ratio']:.2%}")
        print(f"Segments: {stats['segments_count']}")
        
        # 処理済み音声をファイルに保存
        output_file = audio_file.rsplit('.', 1)[0] + '_vad_processed.pcm'
        with open(output_file, 'wb') as f:
            f.write(processed_audio)
        print(f"Processed audio saved to: {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)