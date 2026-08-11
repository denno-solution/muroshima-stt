"""VAD保持区間と単語タイムスタンプ復元(remap)のテスト。"""

import os
import sys
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from services.vad import VadTimeRange, _energy_trim, _identity_ranges  # noqa: E402
from services.word_timestamps import (  # noqa: E402
    build_word_timestamp_columns,
    map_trimmed_to_original,
    normalize_vad_ranges,
    remap_words_to_original,
)


def _ranges():
    """元音声0-10秒のうち [1,3] と [6,8] を保持した場合の区間。"""
    return [
        VadTimeRange(original_start=1.0, original_end=3.0, trimmed_start=0.0, trimmed_end=2.0),
        VadTimeRange(original_start=6.0, original_end=8.0, trimmed_start=2.0, trimmed_end=4.0),
    ]


class TestMapTrimmedToOriginal:
    def test_first_range(self):
        assert map_trimmed_to_original(0.5, _ranges()) == pytest.approx(1.5)

    def test_second_range_with_gap_restored(self):
        # トリム後2.5秒 = 2番目の保持区間の0.5秒目 → 元音声6.5秒
        assert map_trimmed_to_original(2.5, _ranges()) == pytest.approx(6.5)

    def test_boundary_maps_to_previous_range_end(self):
        # 区間境界(2.0)は前の区間の終端として扱う(desktopの許容誤差1e-6と同じ挙動)
        assert map_trimmed_to_original(2.0, _ranges()) == pytest.approx(3.0)

    def test_last_range_end_inclusive(self):
        assert map_trimmed_to_original(4.0, _ranges()) == pytest.approx(8.0)

    def test_out_of_range_returns_none(self):
        assert map_trimmed_to_original(5.0, _ranges()) is None
        assert map_trimmed_to_original(-1.0, _ranges()) is None

    def test_empty_ranges(self):
        assert map_trimmed_to_original(1.0, []) is None


class TestNormalizeVadRanges:
    def test_filters_invalid_and_sorts(self):
        ranges = [
            VadTimeRange(6.0, 8.0, 2.0, 4.0),
            VadTimeRange(1.0, 1.0, 0.0, 2.0),  # original幅ゼロ → 除外
            VadTimeRange(1.0, 3.0, 0.0, 2.0),
            VadTimeRange(float("nan"), 3.0, 0.0, 2.0),  # 非有限 → 除外
        ]
        out = normalize_vad_ranges(ranges)
        assert [(r.trimmed_start, r.trimmed_end) for r in out] == [(0.0, 2.0), (2.0, 4.0)]

    def test_none_and_empty(self):
        assert normalize_vad_ranges(None) == []
        assert normalize_vad_ranges([]) == []


class TestRemapWordsToOriginal:
    def test_remaps_start_and_end(self):
        words = [
            {"text": "こんにちは", "start": 0.2, "end": 1.0, "type": "word"},
            {"text": "世界", "start": 2.2, "end": 3.8, "type": "word"},
        ]
        out = remap_words_to_original(words, _ranges())
        assert out is not None
        assert out[0]["start"] == pytest.approx(1.2)
        assert out[0]["end"] == pytest.approx(2.0)
        assert out[1]["start"] == pytest.approx(6.2)
        assert out[1]["end"] == pytest.approx(7.8)
        # 入力は破壊しない
        assert words[0]["start"] == 0.2

    def test_unmappable_word_keeps_value(self):
        words = [{"text": "外", "start": 9.9, "end": 9.95}]
        out = remap_words_to_original(words, _ranges())
        assert out[0]["start"] == 9.9

    def test_no_ranges_returns_none(self):
        assert remap_words_to_original([{"text": "a", "start": 0.0}], None) is None
        assert remap_words_to_original([{"text": "a", "start": 0.0}], []) is None

    def test_alias_keys(self):
        words = [{"text": "a", "start_time": 0.5, "end_time": 1.5}]
        out = remap_words_to_original(words, _ranges())
        assert out[0]["start_time"] == pytest.approx(1.5)
        assert out[0]["end_time"] == pytest.approx(2.5)


class TestBuildWordTimestampColumns:
    WORDS = [{"text": "テスト", "start": 0.5, "end": 1.0, "type": "word"}]

    def test_no_words(self):
        assert build_word_timestamp_columns(None, vad_applied=True) == (None, None)
        assert build_word_timestamp_columns([], vad_applied=False) == (None, None)

    def test_without_vad_both_columns_equal(self):
        raw, original = build_word_timestamp_columns(self.WORDS, vad_applied=False)
        assert raw == self.WORDS
        assert original == self.WORDS

    def test_with_vad_original_is_remapped(self):
        raw, original = build_word_timestamp_columns(
            self.WORDS, vad_applied=True, vad_ranges=_ranges()
        )
        assert raw[0]["start"] == 0.5  # VAD後基準はそのまま
        assert original[0]["start"] == pytest.approx(1.5)  # 元音声基準へ復元

    def test_with_vad_but_no_ranges(self):
        raw, original = build_word_timestamp_columns(
            self.WORDS, vad_applied=True, vad_ranges=None
        )
        assert raw == self.WORDS
        assert original is None


class TestVadKeptRanges:
    def test_identity_ranges(self):
        rs = _identity_ranges(12.5)
        assert len(rs) == 1
        assert (rs[0].original_start, rs[0].original_end) == (0.0, 12.5)
        assert (rs[0].trimmed_start, rs[0].trimmed_end) == (0.0, 12.5)
        assert _identity_ranges(0.0) == []

    def test_energy_trim_returns_consistent_ranges(self):
        sr = 16000
        # 1秒音声 + 2秒無音 + 1秒音声
        t = np.linspace(0, 1, sr, endpoint=False)
        tone = 0.5 * np.sin(2 * np.pi * 440 * t)
        audio = np.concatenate([tone, np.zeros(2 * sr), tone]).astype(np.float32)

        trimmed, out_sec, ranges = _energy_trim(audio, sr, top_db=30, pad_ms=0)
        assert out_sec < 4.0  # 無音がカットされている
        assert len(ranges) == 2
        # トリム後タイムラインは連続しており、合計が出力長と一致する
        assert ranges[0].trimmed_start == pytest.approx(0.0)
        for prev, cur in zip(ranges, ranges[1:]):
            assert cur.trimmed_start == pytest.approx(prev.trimmed_end)
        assert ranges[-1].trimmed_end == pytest.approx(out_sec, abs=1e-6)
        # 2番目の保持区間は元音声の3秒付近から始まる
        assert ranges[1].original_start == pytest.approx(3.0, abs=0.2)
        # 対応付け: トリム後の2番目区間の中央 → 元音声の対応時刻
        mid = (ranges[1].trimmed_start + ranges[1].trimmed_end) / 2
        mapped = map_trimmed_to_original(mid, ranges)
        expected = ranges[1].original_start + (mid - ranges[1].trimmed_start)
        assert mapped == pytest.approx(expected)
