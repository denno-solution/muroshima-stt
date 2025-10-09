import hashlib
import logging
from pathlib import Path

import librosa
import soundfile as sf

logger = logging.getLogger(__name__)


def md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def should_convert_to_wav(model_name: str) -> bool:
    return model_name in [
        "Google Cloud (Chirp)",
        "Amazon Transcribe",
        "Azure Speech",
    ]


def convert_webm_to_wav(src_path: str, target_sr: int = 16000) -> tuple[str, float]:
    """WebM → WAV 変換し、(wav_path, duration_sec) を返す。
    失敗時は例外を送出する。
    """
    audio_data, sr = librosa.load(src_path, sr=target_sr)
    duration = len(audio_data) / sr
    wav_path = str(Path(src_path).with_suffix('.wav'))
    sf.write(wav_path, audio_data, sr)
    return wav_path, duration


def get_audio_duration(src_path: str, target_sr: int = 16000) -> float:
    """Return duration seconds by decoding audio locally.
    Falls back to 0.0 on error to avoid breaking flows.
    """
    try:
        audio_data, sr = librosa.load(src_path, sr=target_sr)
        return len(audio_data) / sr
    except Exception:
        return 0.0
