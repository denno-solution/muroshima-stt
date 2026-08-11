"""チャンクへの録音内時刻(start_sec/end_sec)割り当て。

単語タイムスタンプ(word_timestamps_*_json)とチャンクテキストを文字位置
ベースで対応付ける近似ロジック。チャンカー(chunker.py)は文単位で空白を
除去して連結するため、空白を除去した正規化空間で位置合わせを行う。

- 時刻の基準は word_timestamps_original_json(VAD前=元音声基準)を優先し、
  無い行のみ word_timestamps_json(VAD後基準)を使う。どちらを使ったかは
  time_basis("original" / "vad")としてチャンクに記録する。
"""

from __future__ import annotations

import json
import logging
import re
from bisect import bisect_left, bisect_right
from typing import Any, Dict, List, Optional, Tuple

from services.word_timestamps import get_word_time_range

logger = logging.getLogger(__name__)

TIME_BASIS_ORIGINAL = "original"  # VAD前(元音声)基準
TIME_BASIS_VAD = "vad"  # VAD後音声基準(元音声の再生位置とズレの可能性あり)

_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """空白類を除去する(チャンカーの文strip・連結と整合させるため)。"""
    return _WS_RE.sub("", text or "")


def parse_words_json(value: Any) -> Optional[List[Dict[str, Any]]]:
    """word_timestamps_*_json 列の値を単語配列として解釈する(寛容)。"""
    if value is None:
        return None
    parsed = value
    if isinstance(value, (str, bytes)):
        raw = value.strip() if isinstance(value, str) else value
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            logger.warning("word_timestampsのJSONパースに失敗したため無視します")
            return None
    if not isinstance(parsed, list):
        return None
    words = [w for w in parsed if isinstance(w, dict)]
    return words or None


def select_timing_words(
    word_timestamps_json: Any,
    word_timestamps_original_json: Any,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """時刻割り当てに使う単語列と基準(time_basis)を決める。

    original(VAD前基準)があればそれを使う。無ければraw(VAD後基準)に
    フォールバックし、基準が元音声とズレうることを "vad" として返す。
    """
    original = parse_words_json(word_timestamps_original_json)
    if original:
        return original, TIME_BASIS_ORIGINAL
    raw = parse_words_json(word_timestamps_json)
    if raw:
        return raw, TIME_BASIS_VAD
    return None, None


def assign_chunk_times(
    transcript: str,
    chunks: List[str],
    words: Optional[List[Dict[str, Any]]],
    chunk_overlap: int = 120,
) -> List[Tuple[Optional[float], Optional[float]]]:
    """各チャンクに録音内時刻範囲(開始秒, 終了秒)を割り当てる。

    - チャンクの位置は「空白除去した本文」内の文字位置で特定する
      (チャンカーは文をstripして連結するため、正規化すれば完全一致する)。
    - 文字位置→時刻は、単語テキストを同様に正規化して連結したときの
      文字オフセットで対応付ける。本文と単語連結の長さが異なる場合
      (クレンジング差分等)は比例スケーリングで近似する。
    - chunk_overlapにはチャンク化時のオーバーラップ文字数を渡す。次チャンクの
      探索開始位置を「前チャンク終端-オーバーラップ」に制限することで、
      同じ文言が繰り返される本文でも手前の同一文字列に誤マッチしない。
    - 対応付けできないチャンクは (None, None)。
    """
    n = len(chunks)
    empty: List[Tuple[Optional[float], Optional[float]]] = [(None, None)] * n
    if not chunks or not words:
        return empty

    t_norm = _normalize(transcript)
    if not t_norm:
        return empty

    # 単語→正規化空間の文字オフセット表(時刻を持つ単語のみentriesに載せる)
    entries: List[Tuple[int, int, float, float]] = []  # (norm_start, norm_end, t_start, t_end)
    total = 0
    for w in words:
        text = _normalize(str(w.get("text", "")))
        length = len(text)
        if length == 0:
            continue  # spacing等は正規化空間に現れない
        start_sec, end_sec = get_word_time_range(w)
        if start_sec is not None:
            if end_sec is None or end_sec < start_sec:
                end_sec = start_sec
            entries.append((total, total + length, float(start_sec), float(end_sec)))
        total += length

    if total == 0 or not entries:
        return empty

    scale = total / len(t_norm)
    entry_starts = [e[0] for e in entries]
    entry_ends = [e[1] for e in entries]

    overlap_bound = max(0, int(chunk_overlap))
    results: List[Tuple[Optional[float], Optional[float]]] = []
    cursor = 0
    for chunk in chunks:
        c_norm = _normalize(chunk)
        if not c_norm:
            results.append((None, None))
            continue
        idx = t_norm.find(c_norm, cursor)
        if idx < 0:
            idx = t_norm.find(c_norm)
        if idx < 0:
            results.append((None, None))
            continue
        # 次チャンクは「このチャンクの終端 - オーバーラップ」以降から始まる
        # (正規化でオーバーラップ部分が縮むことはあっても伸びることはない)
        cursor = max(cursor, idx + len(c_norm) - overlap_bound)

        lo = idx * scale
        hi = (idx + len(c_norm)) * scale
        # [lo, hi) と重なる最初/最後の単語entryを二分探索で求める
        first = bisect_right(entry_ends, lo)
        last = bisect_left(entry_starts, hi) - 1
        if first > last or first >= len(entries) or last < 0:
            results.append((None, None))
            continue
        start_sec = entries[first][2]
        end_sec = max(entries[last][3], start_sec)
        results.append((round(start_sec, 3), round(end_sec, 3)))

    return results
