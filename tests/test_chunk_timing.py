"""チャンクへの録音内時刻割り当て(chunk_timing)のテスト。"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from services.rag.chunk_timing import (  # noqa: E402
    assign_chunk_times,
    parse_words_json,
    select_timing_words,
)
from services.rag.chunker import chunk_text  # noqa: E402


def _make_words(sentences, sec_per_char=0.5, gap=0.0):
    """文のリストから1文字ずつの単語タイムスタンプを合成する。"""
    words = []
    t = 0.0
    for s in sentences:
        for ch in s:
            words.append({"text": ch, "start": round(t, 3), "end": round(t + sec_per_char, 3), "type": "word"})
            t += sec_per_char
        t += gap
    return words


class TestParseWordsJson:
    def test_json_string(self):
        value = json.dumps([{"text": "あ", "start": 0.0, "end": 0.5}])
        words = parse_words_json(value)
        assert words and words[0]["text"] == "あ"

    def test_list_passthrough(self):
        words = parse_words_json([{"text": "あ"}])
        assert words == [{"text": "あ"}]

    def test_invalid_values(self):
        assert parse_words_json(None) is None
        assert parse_words_json("") is None
        assert parse_words_json("not json") is None
        assert parse_words_json("{}") is None
        assert parse_words_json("[]") is None
        assert parse_words_json([1, 2]) is None


class TestSelectTimingWords:
    RAW = json.dumps([{"text": "あ", "start": 0.0}])
    ORIGINAL = json.dumps([{"text": "あ", "start": 10.0}])

    def test_prefers_original(self):
        words, basis = select_timing_words(self.RAW, self.ORIGINAL)
        assert basis == "original"
        assert words[0]["start"] == 10.0

    def test_falls_back_to_raw_as_vad_basis(self):
        words, basis = select_timing_words(self.RAW, None)
        assert basis == "vad"
        assert words[0]["start"] == 0.0

    def test_none_when_neither(self):
        assert select_timing_words(None, None) == (None, None)
        assert select_timing_words("", "not json") == (None, None)


class TestAssignChunkTimes:
    def test_two_chunks_get_monotonic_ranges(self):
        s1 = "最初の文です。" * 10  # 70文字
        s2 = "次の文になります。" * 10  # 90文字
        transcript = s1 + s2
        chunks = list(chunk_text(transcript, chunk_size=80, chunk_overlap=0))
        assert len(chunks) >= 2
        words = _make_words([transcript], sec_per_char=0.1)
        times = assign_chunk_times(transcript, chunks, words, chunk_overlap=0)
        assert len(times) == len(chunks)
        assert times[0][0] == pytest.approx(0.0)
        # 各チャンクの時刻は単調に進む
        for (s_prev, e_prev), (s_cur, e_cur) in zip(times, times[1:]):
            assert s_cur >= s_prev
            assert e_cur >= e_prev
        # 最後のチャンクの終了は全体の長さ(160文字×0.1秒)にほぼ一致
        assert times[-1][1] == pytest.approx(len(transcript) * 0.1, abs=0.2)

    def test_overlap_chunks_are_located_correctly(self):
        transcript = "".join(f"文{i}の内容はここ。" for i in range(40))  # 360文字
        chunks = list(chunk_text(transcript, chunk_size=100, chunk_overlap=30))
        words = _make_words([transcript], sec_per_char=0.2)
        times = assign_chunk_times(transcript, chunks, words, chunk_overlap=30)
        assert all(s is not None and e is not None for s, e in times)
        # オーバーラップがあるため、次チャンクの開始は前チャンクの終了より手前
        assert times[1][0] < times[0][1]
        assert times[1][0] > times[0][0]

    def test_whitespace_between_sentences_is_tolerated(self):
        # チャンカーは文をstripして連結するため、本文に空白・改行があっても
        # 正規化空間で対応付けできる
        transcript = "こんにちは。\n 今日は晴れです。\nありがとう。"
        chunks = list(chunk_text(transcript, chunk_size=12, chunk_overlap=0))
        assert len(chunks) >= 2
        words = _make_words(["こんにちは。", "今日は晴れです。", "ありがとう。"], sec_per_char=0.5)
        times = assign_chunk_times(transcript, chunks, words)
        assert times[0][0] == pytest.approx(0.0)
        assert all(s is not None for s, _ in times)

    def test_words_with_spacing_entries(self):
        # ElevenLabsはtype=spacingの空白トークンを含む。正規化で無視される
        transcript = "Hello world. Second sentence."
        words = []
        t = 0.0
        for token in ["Hello", " ", "world.", " ", "Second", " ", "sentence."]:
            entry = {"text": token, "start": round(t, 2), "end": round(t + 0.3, 2)}
            entry["type"] = "spacing" if token == " " else "word"
            words.append(entry)
            t += 0.3
        chunks = list(chunk_text(transcript, chunk_size=15, chunk_overlap=0))
        times = assign_chunk_times(transcript, chunks, words)
        assert times[0][0] == pytest.approx(0.0)
        assert all(s is not None for s, _ in times)
        # 2番目のチャンク("Second sentence.")は後半の単語時刻になる
        assert times[-1][0] > times[0][0]

    def test_scaling_when_transcript_differs_slightly(self):
        # クレンジングでタグが除去され、本文と単語連結の長さがズレるケース
        transcript = "本文はこれだけ。"  # 8文字
        words = _make_words(["本文はこれだけ。(laughter)"], sec_per_char=1.0)
        times = assign_chunk_times(transcript, [transcript], words)
        s, e = times[0]
        assert s == pytest.approx(0.0)
        assert e is not None and e > 0

    def test_unmatched_chunk_gets_none(self):
        transcript = "実際の本文。"
        words = _make_words([transcript], sec_per_char=0.5)
        times = assign_chunk_times(transcript, ["存在しないチャンク"], words)
        assert times == [(None, None)]

    def test_no_words_or_empty(self):
        assert assign_chunk_times("本文。", ["本文。"], None) == [(None, None)]
        assert assign_chunk_times("本文。", ["本文。"], []) == [(None, None)]
        assert assign_chunk_times("", ["本文。"], [{"text": "x", "start": 0}]) == [(None, None)]

    def test_words_without_times_are_skipped(self):
        transcript = "前半の文。後半の文。"
        words = [
            {"text": "前半の文。", "type": "word"},  # 時刻なし
            {"text": "後半の文。", "start": 5.0, "end": 8.0, "type": "word"},
        ]
        chunks = ["前半の文。", "後半の文。"]
        times = assign_chunk_times(transcript, chunks, words)
        assert times[0] == (None, None)
        assert times[1] == (pytest.approx(5.0), pytest.approx(8.0))
