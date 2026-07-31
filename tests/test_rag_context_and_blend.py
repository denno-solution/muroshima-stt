import os
import sys
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from services.rag.context_builder import (  # noqa: E402
    _join_with_overlap,
    _merge_chunk_windows,
    build_context_docs,
)
from services.rag.search_service import blend_scores  # noqa: E402
from services.rag.reconcile import normalize_to_jst_date  # noqa: E402
from datetime import datetime, timezone  # noqa: E402


class TestBlendScoresRRF:
    def test_vector_only(self):
        vec = [{"source": "audio", "chunk_id": 1}, {"source": "audio", "chunk_id": 2}]
        out = blend_scores(vec, [], alpha=1.0)
        assert [r["chunk_id"] for r in out] == [1, 2]
        assert out[0]["score"] == pytest.approx(1.0)  # 単独1位=1.0スケール

    def test_union_and_boost_for_both_lists(self):
        vec = [{"source": "audio", "chunk_id": 1}, {"source": "audio", "chunk_id": 2}]
        fts = [{"source": "audio", "chunk_id": 2}, {"source": "audio", "chunk_id": 3}]
        out = blend_scores(vec, fts, alpha=0.5)
        by_id = {r["chunk_id"]: r for r in out}
        # 両方に出る2が最上位
        assert out[0]["chunk_id"] == 2
        assert by_id[2]["score"] > by_id[1]["score"]
        assert set(by_id) == {1, 2, 3}

    def test_alpha_zero_ignores_vector(self):
        vec = [{"source": "audio", "chunk_id": 1}]
        fts = [{"source": "audio", "chunk_id": 2}]
        out = blend_scores(vec, fts, alpha=0.0)
        assert out[0]["chunk_id"] == 2


class TestJoinWithOverlap:
    def test_removes_overlap(self):
        assert _join_with_overlap("abcdef", "defghi") == "abcdefghi"

    def test_no_overlap(self):
        assert _join_with_overlap("abc", "xyz") == "abcxyz"


class TestMergeChunkWindows:
    CHUNKS = [(0, "AAA"), (1, "BBB"), (2, "CCC"), (3, "DDD"), (4, "EEE")]

    def test_single_hit_with_neighbors(self):
        out = _merge_chunk_windows(self.CHUNKS, {2}, window=1)
        assert out == "BBBCCCDDD"

    def test_disjoint_hits_marked_with_gap(self):
        out = _merge_chunk_windows(self.CHUNKS, {0, 4}, window=0)
        assert out == "AAA\n（…中略…）\nEEE"


@pytest.fixture()
def mini_db():
    engine = create_engine("sqlite://")
    with engine.begin() as c:
        c.execute(text(
            "CREATE TABLE audio_transcriptions (id INTEGER PRIMARY KEY, transcript TEXT)"
        ))
        c.execute(text(
            "CREATE TABLE audio_transcription_chunks ("
            "id INTEGER PRIMARY KEY, transcription_id INTEGER, chunk_index INTEGER, chunk_text TEXT)"
        ))
        c.execute(text("INSERT INTO audio_transcriptions VALUES (1, '短い録音の全文です。')"))
        long_text = "長い録音。" * 2000  # 10000文字
        c.execute(text("INSERT INTO audio_transcriptions VALUES (2, :t)"), {"t": long_text})
        for i in range(5):
            c.execute(
                text("INSERT INTO audio_transcription_chunks VALUES (:id, 2, :idx, :t)"),
                {"id": 100 + i, "idx": i, "t": f"チャンク{i}の本文。" * 10},
            )
    Session = sessionmaker(bind=engine)
    db = Session()
    yield db
    db.close()


def _hit(tid, chunk_index, score, **kw):
    return {
        "source": "audio",
        "transcription_id": tid,
        "chunk_index": chunk_index,
        "title": f"rec{tid}.mp3",
        "recorded_date": kw.get("recorded_date", "2026-07-01"),
        "tags": None,
        "score": score,
    }


class TestBuildContextDocs:
    def test_short_doc_gets_full_text(self, mini_db):
        docs = build_context_docs(
            mini_db, [_hit(1, 0, 0.9)], max_docs=3, max_chars=40000, whole_doc_threshold=4000
        )
        assert len(docs) == 1
        assert docs[0].is_full_text
        assert docs[0].text == "短い録音の全文です。"

    def test_long_doc_uses_hit_windows(self, mini_db):
        docs = build_context_docs(
            mini_db, [_hit(2, 2, 0.8)], max_docs=3, max_chars=40000, whole_doc_threshold=4000
        )
        assert len(docs) == 1
        assert not docs[0].is_full_text
        assert "チャンク1の本文" in docs[0].text  # 前後チャンクも含む
        assert "チャンク3の本文" in docs[0].text
        assert "チャンク0の本文" not in docs[0].text

    def test_grouping_and_score_order(self, mini_db):
        hits = [_hit(1, 0, 0.5), _hit(2, 1, 0.9), _hit(2, 3, 0.7)]
        docs = build_context_docs(
            mini_db, hits, max_docs=3, max_chars=40000, whole_doc_threshold=4000
        )
        assert [d.transcription_id for d in docs] == [2, 1]
        assert docs[0].hit_count == 2

    def test_date_order(self, mini_db):
        hits = [
            _hit(1, 0, 0.9, recorded_date="2026-06-01"),
            _hit(2, 1, 0.5, recorded_date="2026-07-01"),
        ]
        docs = build_context_docs(
            mini_db, hits, max_docs=3, max_chars=40000, whole_doc_threshold=4000, order="date"
        )
        assert [d.transcription_id for d in docs] == [2, 1]

    def test_max_docs_budget(self, mini_db):
        hits = [_hit(1, 0, 0.9), _hit(2, 1, 0.8)]
        docs = build_context_docs(
            mini_db, hits, max_docs=1, max_chars=40000, whole_doc_threshold=4000
        )
        assert len(docs) == 1


class TestNormalizeToJstDate:
    def test_naive_string_treated_as_utc(self):
        # UTC 23:22 → JST 翌日08:22
        assert normalize_to_jst_date("2025-07-30 23:22:21.308854") == "2025-07-31"

    def test_rfc3339_with_nanoseconds(self):
        assert normalize_to_jst_date("2026-07-30T08:31:12.906558100+00:00") == "2026-07-30"

    def test_z_suffix(self):
        assert normalize_to_jst_date("2026-02-19T06:42:05.644Z") == "2026-02-19"

    def test_utc_evening_rolls_to_next_jst_day(self):
        assert normalize_to_jst_date("2026-07-30T16:00:00+00:00") == "2026-07-31"

    def test_datetime_object(self):
        dt = datetime(2026, 7, 30, 23, 0, 0)  # naive→UTC扱い
        assert normalize_to_jst_date(dt) == "2026-07-31"

    def test_aware_datetime(self):
        dt = datetime(2026, 7, 30, 12, 0, 0, tzinfo=timezone.utc)
        assert normalize_to_jst_date(dt) == "2026-07-30"

    def test_none_and_garbage(self):
        assert normalize_to_jst_date(None) is None
        assert normalize_to_jst_date("") is None
        assert normalize_to_jst_date("not a date") is None


class TestPerDocCap:
    def test_cap_limits_each_doc(self, mini_db):
        # 期間要約モード: 多数の録音を1件あたり少ない文字数で読む
        docs = build_context_docs(
            mini_db,
            [_hit(2, 1, 0.9)],
            max_docs=30,
            max_chars=40000,
            whole_doc_threshold=4000,
            per_doc_cap=100,
        )
        assert len(docs) == 1
        assert docs[0].truncated
        assert len(docs[0].text) <= 100 + len("\n（文字数上限のため以下省略）")

    def test_default_behavior_unchanged_without_cap(self, mini_db):
        docs = build_context_docs(
            mini_db,
            [_hit(1, 0, 0.9)],
            max_docs=3,
            max_chars=40000,
            whole_doc_threshold=4000,
        )
        assert docs[0].text == "短い録音の全文です。"
        assert not docs[0].truncated
