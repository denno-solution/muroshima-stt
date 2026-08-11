"""単語タイムスタンプ(word timestamps)の保存用ユーティリティ。

stt-desktop と同じスキーマで `word_timestamps_json`(STTが返したまま=VAD後
音声基準) と `word_timestamps_original_json`(VAD前=元音声基準に復元) の
両方を保存するための変換を提供する。

- 復元ロジックは stt-desktop の remap_word_timestamps_to_original
  (src-tauri/src/transcript.rs) と同等。
- VADを通らない経路ではSTTのタイムスタンプ=元音声基準なので両者は同値。
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from services.vad import VadTimeRange

logger = logging.getLogger(__name__)

# stt-desktop と同じキー候補(先頭が優先)
_START_KEYS = ("start", "start_time", "startTime", "offset_start")
_END_KEYS = ("end", "end_time", "endTime", "offset_end")


def _to_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        f = float(value)
    elif isinstance(value, str):
        try:
            f = float(value)
        except ValueError:
            return None
    else:
        return None
    return f if math.isfinite(f) else None


def _get_time_value(word: Dict[str, Any], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        if key in word:
            seconds = _to_float(word[key])
            if seconds is not None:
                return seconds
    return None


def _set_time_value(word: Dict[str, Any], keys: Sequence[str], value: float) -> None:
    for key in keys:
        if key in word:
            word[key] = value
            return
    word[keys[0]] = value


def get_word_time_range(
    word: Dict[str, Any],
) -> Tuple[Optional[float], Optional[float]]:
    """単語オブジェクトから (start秒, end秒) を取り出す(キー表記ゆれ対応)。"""
    return _get_time_value(word, _START_KEYS), _get_time_value(word, _END_KEYS)


def normalize_vad_ranges(
    ranges: Optional[Sequence[VadTimeRange]],
) -> List[VadTimeRange]:
    """不正な区間を除外し、トリム後時刻の昇順に整列する。"""
    if not ranges:
        return []
    out = [
        r
        for r in ranges
        if math.isfinite(r.original_start)
        and math.isfinite(r.original_end)
        and math.isfinite(r.trimmed_start)
        and math.isfinite(r.trimmed_end)
        and r.original_end > r.original_start
        and r.trimmed_end > r.trimmed_start
    ]
    out.sort(key=lambda r: r.trimmed_start)
    return out


def map_trimmed_to_original(
    seconds: float, ranges: Sequence[VadTimeRange]
) -> Optional[float]:
    """トリム後音声の時刻(秒)を元音声の時刻へ写像する。範囲外はNone。"""
    if not math.isfinite(seconds) or not ranges:
        return None
    for idx, r in enumerate(ranges):
        is_last = idx + 1 == len(ranges)
        if is_last:
            in_range = r.trimmed_start - 1e-6 <= seconds <= r.trimmed_end + 1e-6
        else:
            in_range = r.trimmed_start - 1e-6 <= seconds < r.trimmed_end + 1e-6
        if in_range:
            max_delta = max(r.trimmed_end - r.trimmed_start, 0.0)
            delta = min(max(seconds - r.trimmed_start, 0.0), max_delta)
            return r.original_start + delta
    return None


def remap_words_to_original(
    words: Optional[List[Dict[str, Any]]],
    vad_ranges: Optional[Sequence[VadTimeRange]],
) -> Optional[List[Dict[str, Any]]]:
    """VAD後基準の単語タイムスタンプを元音声基準へ復元する。

    stt-desktop の remap_word_timestamps_to_original と同等。復元できない
    単語は元の値を保持する。区間情報がない場合はNone(復元不能)。
    """
    ranges = normalize_vad_ranges(vad_ranges)
    if not ranges or not words:
        return None

    mapped: List[Dict[str, Any]] = []
    for word in words:
        copied = dict(word) if isinstance(word, dict) else word
        if isinstance(copied, dict):
            start_sec = _get_time_value(copied, _START_KEYS)
            if start_sec is not None:
                mapped_start = map_trimmed_to_original(start_sec, ranges)
                if mapped_start is not None:
                    _set_time_value(copied, _START_KEYS, mapped_start)
            end_sec = _get_time_value(copied, _END_KEYS)
            if end_sec is not None:
                mapped_end = map_trimmed_to_original(end_sec, ranges)
                if mapped_end is not None:
                    _set_time_value(copied, _END_KEYS, mapped_end)
        mapped.append(copied)
    return mapped


def build_word_timestamp_columns(
    words: Optional[List[Dict[str, Any]]],
    *,
    vad_applied: bool,
    vad_ranges: Optional[Sequence[VadTimeRange]] = None,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[List[Dict[str, Any]]]]:
    """保存用の (word_timestamps_json, word_timestamps_original_json) を作る。

    - vad_applied=False: STT入力=元音声なので両カラムに同値を保存する。
    - vad_applied=True: rawはそのまま、originalは保持区間から復元する。
      区間情報が無く復元できない場合のみ original は None。
    """
    if not words:
        return None, None
    if not vad_applied:
        return words, words
    original = remap_words_to_original(words, vad_ranges)
    if original is None:
        logger.warning(
            "VAD保持区間が無いため元音声基準のタイムスタンプを復元できませんでした"
        )
    return words, original
