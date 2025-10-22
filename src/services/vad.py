from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import librosa

logger = logging.getLogger(__name__)


@dataclass
class VADResult:
    applied: bool
    method: str
    input_path: str
    output_path: str
    orig_sec: float
    out_sec: float


def _to_int16_pcm(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)


def _energy_trim(audio: np.ndarray, sr: int, top_db: int = 30, pad_ms: int = 150) -> Tuple[np.ndarray, float]:
    """librosaのエネルギーベースで非音声（無音）をカットする簡易版。
    webrtcvadが使えない環境のフォールバックとして使用。
    """
    intervals = librosa.effects.split(audio, top_db=top_db)
    if intervals.size == 0:
        return audio, float(len(audio) / sr)
    pad = int(sr * pad_ms / 1000)
    pieces: List[np.ndarray] = []
    for start, end in intervals:
        s = max(0, start - pad)
        e = min(len(audio), end + pad)
        pieces.append(audio[s:e])
    out = np.concatenate(pieces) if pieces else audio
    return out, float(len(out) / sr)


def trim_non_speech(
    input_path: str,
    enabled: bool = True,
    aggressiveness: int = 2,
    frame_ms: int = 30,
    pad_ms: int = 150,
    min_out_ms: int = 500,
    target_sr: int = 16000,
) -> VADResult:
    """音声ファイルから人声以外の区間（無音/雑音）をカットしてWAVを出力。

    - 可能なら`webrtcvad`で音声区間検出、失敗時はlibrosaのエネルギートリムにフォールバック。
    - 出力は16kHz mono WAV（ElevenLabs等のSTTが問題なく受け付ける一般的設定）。

    Returns: VADResult
    """
    input_path = str(input_path)
    orig_audio, orig_sr = librosa.load(input_path, sr=None, mono=True)
    orig_sec = float(len(orig_audio) / orig_sr if orig_sr else 0.0)

    out_dir = Path(input_path).parent
    out_path = str(Path(input_path).with_suffix("") ) + "_vad.wav"

    if not enabled:
        # 変換のみ: STTの互換性を維持するため16kHzに変換して保存
        audio_16k = librosa.resample(orig_audio, orig_sr=orig_sr, target_sr=target_sr) if orig_sr != target_sr else orig_audio
        sf.write(out_path, audio_16k, target_sr)
        return VADResult(False, "none", input_path, out_path, orig_sec, float(len(audio_16k)/target_sr))

    try:
        import webrtcvad  # type: ignore

        vad = webrtcvad.Vad(int(aggressiveness))
        # VADは16k, mono, 16bit PCMで動作させる
        audio_16k = librosa.resample(orig_audio, orig_sr=orig_sr, target_sr=target_sr) if orig_sr != target_sr else orig_audio
        pcm16 = _to_int16_pcm(audio_16k)

        frame_len = int(target_sr * frame_ms / 1000)
        if frame_len <= 0:
            frame_len = int(target_sr * 0.03)

        n_frames = len(pcm16) // frame_len
        if n_frames == 0:
            # きわめて短いファイルはスルー
            sf.write(out_path, audio_16k, target_sr)
            return VADResult(True, "webrtcvad", input_path, out_path, orig_sec, float(len(audio_16k)/target_sr))

        speech_flags = []
        for i in range(n_frames):
            frame = pcm16[i * frame_len : (i + 1) * frame_len].tobytes()
            try:
                speech_flags.append(vad.is_speech(frame, sample_rate=target_sr))
            except Exception:
                # 異常フレームは無音扱い
                speech_flags.append(False)

        # 簡易スムージング（前後pad_msだけ膨らませる）
        pad_frames = max(1, int(pad_ms / frame_ms))
        speech_arr = np.array(speech_flags, dtype=bool)
        if speech_arr.any():
            # 膨張（dilation）
            kernel = np.ones(2 * pad_frames + 1, dtype=int)
            smoothed = np.convolve(speech_arr.astype(int), kernel, mode="same") > 0
        else:
            smoothed = speech_arr

        # 区間化
        segments: List[Tuple[int, int]] = []
        in_seg = False
        seg_start = 0
        for i, flag in enumerate(smoothed):
            if flag and not in_seg:
                in_seg = True
                seg_start = i
            elif not flag and in_seg:
                in_seg = False
                segments.append((seg_start * frame_len, i * frame_len))
        if in_seg:
            segments.append((seg_start * frame_len, n_frames * frame_len))

        if not segments:
            # すべて非音声と判定された場合は原音を書き出す
            sf.write(out_path, audio_16k, target_sr)
            return VADResult(True, "webrtcvad_empty", input_path, out_path, orig_sec, float(len(audio_16k)/target_sr))

        pieces = [audio_16k[s:e] for s, e in segments]
        trimmed = np.concatenate(pieces) if pieces else audio_16k
        out_sec = float(len(trimmed) / target_sr)

        # 短すぎるとSTTが不安定になるのでフォールバック
        if out_sec * 1000 < min_out_ms:
            sf.write(out_path, audio_16k, target_sr)
            return VADResult(True, "webrtcvad_short_fallback", input_path, out_path, orig_sec, float(len(audio_16k)/target_sr))

        sf.write(out_path, trimmed, target_sr)
        return VADResult(True, "webrtcvad", input_path, out_path, orig_sec, out_sec)

    except Exception as e:
        logger.warning(f"webrtcvadが使用できないためエネルギートリムにフォールバックします: {e}")
        # フォールバック（エネルギーベース）
        audio_16k = librosa.resample(orig_audio, orig_sr=orig_sr, target_sr=target_sr) if orig_sr != target_sr else orig_audio
        trimmed, out_sec = _energy_trim(audio_16k, target_sr, top_db=30, pad_ms=pad_ms)
        if out_sec * 1000 < min_out_ms:
            trimmed = audio_16k
            out_sec = float(len(trimmed) / target_sr)
            method = "energy_short_fallback"
        else:
            method = "energy"
        sf.write(out_path, trimmed, target_sr)
        return VADResult(True, method, input_path, out_path, orig_sec, out_sec)

