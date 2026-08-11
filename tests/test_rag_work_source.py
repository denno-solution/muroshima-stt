"""workソース(業務記録)の検索絞り込みテスト。

業務記録はceo_transcriptionsにtags='業務記録'で保存される(デスクトップ版PR #17)。
検索時のみタグで分岐し、既存行(tags NULL)や他タグの行は従来通り
社長音声(ceo)として検索されることを保証する。
"""

import os
import sys
from datetime import date
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402
from sqlalchemy import create_engine, event, text  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from services.rag.context_builder import _PARENT_TABLES, build_context_docs  # noqa: E402
from services.rag.prompt_builder import (  # noqa: E402
    _SOURCE_LABELS,
    build_system_prompt,
)
from services.rag.search_service import (  # noqa: E402
    SearchFilters,
    SearchService,
    VALID_SOURCES,
    WORK_RECORD_TAG,
)
from services.rag.tokenizer import fts_query_any, to_fts_text  # noqa: E402

# (id, title, tags, recorded_date, transcript)
_CEO_ROWS = [
    (1, "既存メモ", None, "2026-08-01", "既存データ。在庫の確認をした。"),
    (2, "社長メモ", "社長音声", "2026-08-02", "社長音声。在庫の相談をした。"),
    (3, "業務記録1", WORK_RECORD_TAG, "2026-08-03", "業務記録。在庫の棚卸しをした。"),
    (4, "会議メモ", "会議", "2026-08-04", "他タグ。在庫の報告をした。"),
    (5, "空タグ", "", "2026-08-05", "空タグ。在庫の連絡をした。"),
]


@pytest.fixture()
def db():
    engine = create_engine("sqlite://")

    @event.listens_for(engine, "connect")
    def _register_vector_fns(dbapi_conn, _record):
        # libsql固有のベクトル関数をテスト用に代替する
        # (embedding列に格納した数値文字列をそのまま距離として返す)
        dbapi_conn.create_function("vector32", 1, lambda s: s)
        dbapi_conn.create_function(
            "vector_distance_cos", 2, lambda emb, _q: float(emb)
        )

    with engine.begin() as c:
        c.execute(text(
            "CREATE TABLE audio_transcriptions ("
            "id INTEGER PRIMARY KEY, file_path TEXT, transcript TEXT, tags TEXT, "
            "recorded_date TEXT, created_at TEXT, duration_seconds REAL)"
        ))
        c.execute(text(
            "CREATE TABLE ceo_transcriptions ("
            "id INTEGER PRIMARY KEY, title TEXT, file_path TEXT, transcript TEXT, "
            "tags TEXT, recorded_date TEXT, created_at TEXT, duration_seconds REAL)"
        ))
        for tbl in ("audio_transcription_chunks", "ceo_transcription_chunks"):
            c.execute(text(
                f"CREATE TABLE {tbl} ("
                "id INTEGER PRIMARY KEY, transcription_id INTEGER, chunk_index INTEGER, "
                "chunk_text TEXT, embedding TEXT, start_sec REAL, end_sec REAL, "
                "time_basis TEXT)"
            ))
        c.execute(text("CREATE VIRTUAL TABLE rag_fts_audio USING fts5(tokens)"))
        c.execute(text("CREATE VIRTUAL TABLE rag_fts_ceo USING fts5(tokens)"))

        c.execute(
            text(
                "INSERT INTO audio_transcriptions VALUES "
                "(1, 'field.mp3', '現場録音。在庫の話。', NULL, '2026-08-01', NULL, 10.0)"
            )
        )
        c.execute(text(
            "INSERT INTO audio_transcription_chunks VALUES "
            "(1, 1, 0, '現場録音。在庫の話。', '0.1', NULL, NULL, NULL)"
        ))
        c.execute(
            text("INSERT INTO rag_fts_audio(rowid, tokens) VALUES (1, :tokens)"),
            {"tokens": to_fts_text("現場録音。在庫の話。")},
        )

        for tid, title, tags, rec_date, transcript in _CEO_ROWS:
            c.execute(
                text(
                    "INSERT INTO ceo_transcriptions VALUES "
                    "(:id, :title, NULL, :transcript, :tags, :rec_date, NULL, 10.0)"
                ),
                {"id": tid, "title": title, "tags": tags,
                 "rec_date": rec_date, "transcript": transcript},
            )
            chunk_id = tid * 10
            c.execute(
                text(
                    "INSERT INTO ceo_transcription_chunks VALUES "
                    "(:cid, :tid, 0, :text, :emb, NULL, NULL, NULL)"
                ),
                {"cid": chunk_id, "tid": tid, "text": transcript,
                 "emb": str(0.1 * tid)},
            )
            c.execute(
                text("INSERT INTO rag_fts_ceo(rowid, tokens) VALUES (:cid, :tokens)"),
                {"cid": chunk_id, "tokens": to_fts_text(transcript)},
            )

    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def _tids(rows):
    return sorted(int(r["transcription_id"]) for r in rows)


class TestWorkSourceSplit:
    """work = tags='業務記録' のみ / ceo = それ以外(NULL・他タグ・空文字含む)。"""

    def test_valid_sources(self):
        assert VALID_SOURCES == ("audio", "ceo", "work")

    def test_browse_recent_work_only_hits_work_tag(self, db):
        rows = SearchService().browse_recent(
            db, SearchFilters(sources=("work",)), max_recordings=10
        )
        assert _tids(rows) == [3]
        assert rows[0]["source"] == "work"
        assert rows[0]["tags"] == WORK_RECORD_TAG

    def test_browse_recent_ceo_excludes_work_and_keeps_null(self, db):
        rows = SearchService().browse_recent(
            db, SearchFilters(sources=("ceo",)), max_recordings=10
        )
        # NULL(既存データ)・社長音声・他タグ・空文字はceo扱いのまま
        assert _tids(rows) == [1, 2, 4, 5]
        assert all(r["source"] == "ceo" for r in rows)

    def test_count_recordings_split(self, db):
        svc = SearchService()
        assert svc.count_recordings(db, SearchFilters(sources=("work",))) == 1
        assert svc.count_recordings(db, SearchFilters(sources=("ceo",))) == 4
        assert svc.count_recordings(db, SearchFilters(sources=("ceo", "work"))) == 5

    def test_keyword_search_split(self, db):
        svc = SearchService()
        q = fts_query_any("在庫")
        work = svc.keyword_search(db, q, SearchFilters(sources=("work",)), k=10)
        ceo = svc.keyword_search(db, q, SearchFilters(sources=("ceo",)), k=10)
        assert _tids(work) == [3]
        assert _tids(ceo) == [1, 2, 4, 5]

    def test_vector_search_split(self, db):
        svc = SearchService()
        qvec = [0.0] * 4
        work = svc.vector_search(db, qvec, SearchFilters(sources=("work",)), k=10)
        ceo = svc.vector_search(db, qvec, SearchFilters(sources=("ceo",)), k=10)
        assert _tids(work) == [3]
        assert _tids(ceo) == [1, 2, 4, 5]

    def test_three_source_union_covers_all_rows(self, db):
        svc = SearchService()
        q = fts_query_any("在庫")
        rows = svc.keyword_search(
            db, q, SearchFilters(sources=("audio", "ceo", "work")), k=10
        )
        assert {(r["source"], int(r["transcription_id"])) for r in rows} == {
            ("audio", 1), ("ceo", 1), ("ceo", 2), ("work", 3), ("ceo", 4), ("ceo", 5),
        }

    def test_date_filter_applies_to_work(self, db):
        svc = SearchService()
        in_range = SearchFilters(
            date_from=date(2026, 8, 3), date_to=date(2026, 8, 3), sources=("work",)
        )
        out_of_range = SearchFilters(
            date_from=date(2026, 8, 4), date_to=date(2026, 8, 5), sources=("work",)
        )
        assert _tids(svc.browse_recent(db, in_range, 10)) == [3]
        assert svc.browse_recent(db, out_of_range, 10) == []

    def test_corpus_stats_split(self, db):
        stats = SearchService().corpus_stats(db)
        assert stats["work"]["count"] == 1
        assert stats["ceo"]["count"] == 4
        assert stats["audio"]["count"] == 1
        assert stats["work"]["latest"] == "2026-08-03"
        assert stats["ceo"]["latest"] == "2026-08-05"

    def test_available_date_range_split(self, db):
        svc = SearchService()
        assert svc.available_date_range(db, ("work",)) == ("2026-08-03", "2026-08-03")
        assert svc.available_date_range(db, ("ceo",)) == ("2026-08-01", "2026-08-05")


class TestContextBuilderWork:
    def test_parent_tables_share_ceo_tables(self):
        assert _PARENT_TABLES["work"] == ("ceo_transcriptions", "ceo_transcription_chunks")

    def test_build_context_docs_for_work_hit(self, db):
        hits = [{
            "source": "work",
            "transcription_id": 3,
            "title": "業務記録1",
            "recorded_date": "2026-08-03",
            "tags": WORK_RECORD_TAG,
            "score": 1.0,
            "chunk_id": 30,
            "chunk_index": 0,
            "chunk_text": "業務記録。在庫の棚卸しをした。",
        }]
        docs = build_context_docs(db, hits, max_docs=5, max_chars=10000)
        assert len(docs) == 1
        assert docs[0].source == "work"
        assert "棚卸し" in docs[0].text


class TestPromptBuilderWork:
    def test_source_label(self):
        assert _SOURCE_LABELS["work"] == "業務記録"

    def test_system_prompt_work_only(self):
        prompt = build_system_prompt(today=date(2026, 8, 11), corpus=("work",))
        assert "業務記録" in prompt
        assert "社長音声DB" not in prompt

    def test_system_prompt_multi_source_lists_all(self):
        prompt = build_system_prompt(
            today=date(2026, 8, 11), corpus=("audio", "ceo", "work")
        )
        assert "音声DB" in prompt
        assert "社長音声DB" in prompt
        assert "業務記録" in prompt

    def test_system_prompt_accepts_plain_string(self):
        # 既存の呼び出し形式(corpus="ceo")の後方互換
        prompt = build_system_prompt(today=date(2026, 8, 11), corpus="ceo")
        assert "社長音声DB" in prompt

    def test_system_prompt_unknown_falls_back_to_audio(self):
        prompt = build_system_prompt(today=date(2026, 8, 11), corpus=())
        assert "音声DB" in prompt
