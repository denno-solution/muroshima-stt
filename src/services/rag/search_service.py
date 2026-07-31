"""チャンク検索の実行層。

Phase 2(agentic search)でLLMのツールとしてそのまま公開できるよう、
「フィルタ付きベクトル検索」「フィルタ付きキーワード検索」「期間ブラウズ」を
独立した操作として提供する。日付はSQLのWHERE句で絞り込む(事後フィルタ廃止)。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

import json

from sqlalchemy import text
from sqlalchemy.orm import Session

from models import RAG_FTS_TABLES
from services.rag.tokenizer import fts_query_any

# ソース種別ごとのテーブル定義
_SOURCES: Dict[str, Dict[str, str]] = {
    "audio": {
        "chunks": "audio_transcription_chunks",
        "parent": "audio_transcriptions",
        "fts": RAG_FTS_TABLES["audio"],
        "title": "trans.file_path",
    },
    "ceo": {
        "chunks": "ceo_transcription_chunks",
        "parent": "ceo_transcriptions",
        "fts": RAG_FTS_TABLES["ceo"],
        "title": "COALESCE(NULLIF(trans.title, ''), trans.file_path)",
    },
}

VALID_SOURCES = tuple(_SOURCES.keys())


@dataclass(frozen=True)
class SearchFilters:
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    sources: Sequence[str] = ("audio",)

    def date_params(self) -> Dict[str, Optional[str]]:
        return {
            "date_from": self.date_from.isoformat() if self.date_from else None,
            "date_to": self.date_to.isoformat() if self.date_to else None,
        }


_DATE_WHERE = (
    " AND (:date_from IS NULL OR trans.recorded_date >= :date_from)"
    " AND (:date_to IS NULL OR trans.recorded_date <= :date_to)"
)


def _select_cols(cfg: Dict[str, str], source: str) -> str:
    return (
        f"chunk.id AS chunk_id, chunk.chunk_index AS chunk_index, chunk.chunk_text AS chunk_text, "
        f"trans.id AS transcription_id, {cfg['title']} AS title, trans.tags AS tags, "
        f"trans.recorded_date AS recorded_date, trans.created_at AS recorded_at, "
        f"trans.duration_seconds AS duration, '{source}' AS source"
    )


RRF_K = 60  # Reciprocal Rank Fusion の定数(業界標準値)


def blend_scores(
    vec_rows: List[Dict],
    fts_rows: List[Dict],
    alpha: float,
) -> List[Dict]:
    """重み付きRRF(Reciprocal Rank Fusion)でベクトル/キーワード結果を融合する。

    スコアの絶対値ではなく順位のみを使うため、コサイン距離とBM25という
    尺度の異なるスコアを直接混ぜるキャリブレーション問題が発生しない。
    alpha: ベクトル側の重み(0.0〜1.0)。
    同一チャンクは (source, chunk_id) で同一視する。入力は関連度順であること。
    """
    merged: Dict[Tuple[str, int], Dict] = {}

    def _rrf(rank: int) -> float:
        return 1.0 / (RRF_K + rank + 1)

    for rank, r in enumerate(vec_rows):
        key = (r["source"], int(r["chunk_id"]))
        rec = dict(r)
        rec["score_vector"] = _rrf(rank)
        rec["score_fts"] = 0.0
        merged[key] = rec
    for rank, r in enumerate(fts_rows):
        key = (r["source"], int(r["chunk_id"]))
        if key in merged:
            merged[key]["score_fts"] = _rrf(rank)
        else:
            rec = dict(r)
            rec["score_vector"] = 0.0
            rec["score_fts"] = _rrf(rank)
            merged[key] = rec

    results = list(merged.values())
    # RRFの生値は小さいため、単独1位(1/(K+1))を1.0とするスケールに直す
    scale = 1.0 / _rrf(0)
    for rec in results:
        rec["score"] = scale * (
            alpha * rec["score_vector"] + (1.0 - alpha) * rec["score_fts"]
        )
        rec["score_vector"] = scale * rec["score_vector"]
        rec["score_fts"] = scale * rec["score_fts"]
        rec.pop("distance", None)
        rec.pop("bm25", None)
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


class SearchService:
    """libSQL上のチャンク検索。全メソッドが日付フィルタをSQLで適用する。"""

    def vector_search(
        self,
        db: Session,
        query_vector: List[float],
        filters: SearchFilters,
        k: int,
    ) -> List[Dict]:
        """ベクトル検索(全走査+WHERE)。

        vector_top_k索引はWHERE句と併用できず日付絞り込みで取りこぼすため、
        全走査で正確に上位kを取る。現在の規模(数千チャンク)では十分高速。
        """
        qvec = json.dumps(query_vector)
        rows: List[Dict] = []
        for source in filters.sources:
            cfg = _SOURCES[source]
            stmt = text(
                f"SELECT {_select_cols(cfg, source)}, "
                f"vector_distance_cos(chunk.embedding, vector32(:qvec)) AS distance "
                f"FROM {cfg['chunks']} AS chunk "
                f"JOIN {cfg['parent']} AS trans ON trans.id = chunk.transcription_id "
                f"WHERE chunk.embedding IS NOT NULL{_DATE_WHERE} "
                f"ORDER BY distance ASC LIMIT :k"
            )
            params = {"qvec": qvec, "k": k, **filters.date_params()}
            rows.extend(dict(r) for r in db.execute(stmt, params).mappings().all())
        rows.sort(key=lambda r: float(r.get("distance") or 0.0))
        return rows[:k]

    def keyword_search(
        self,
        db: Session,
        match_query: str,
        filters: SearchFilters,
        k: int,
    ) -> List[Dict]:
        """FTS5(バイグラム索引)によるキーワード検索。match_queryはFTS5構文。"""
        rows: List[Dict] = []
        for source in filters.sources:
            cfg = _SOURCES[source]
            fts = cfg["fts"]
            stmt = text(
                f"SELECT {_select_cols(cfg, source)}, bm25({fts}) AS bm25 "
                f"FROM {fts} "
                f"JOIN {cfg['chunks']} AS chunk ON chunk.id = {fts}.rowid "
                f"JOIN {cfg['parent']} AS trans ON trans.id = chunk.transcription_id "
                f"WHERE {fts} MATCH :q{_DATE_WHERE} "
                f"ORDER BY bm25 LIMIT :k"
            )
            params = {"q": match_query, "k": k, **filters.date_params()}
            rows.extend(dict(r) for r in db.execute(stmt, params).mappings().all())
        rows.sort(key=lambda r: float(r.get("bm25") or 0.0))
        return rows[:k]

    def hybrid_search(
        self,
        db: Session,
        query_text: str,
        query_vector: Optional[List[float]],
        filters: SearchFilters,
        k: int,
        alpha: float,
        match_query: Optional[str] = None,
    ) -> List[Dict]:
        """ベクトル×キーワードのハイブリッド検索。

        match_queryを渡すとFTS側はそれを使う(内容語のみ+同義語展開など、
        呼び出し側で組み立てたクエリを注入できる)。省略時は全文からOR構築。
        """
        cand_k = max(k, k * 3)
        vec_rows = (
            self.vector_search(db, query_vector, filters, cand_k)
            if query_vector
            else []
        )
        match_q = match_query if match_query is not None else fts_query_any(query_text)
        fts_rows = (
            self.keyword_search(db, match_q, filters, cand_k) if match_q else []
        )
        return blend_scores(vec_rows, fts_rows, alpha)[:k]

    def count_recordings(self, db: Session, filters: SearchFilters) -> int:
        """期間内の録音件数(browse時の網羅率表示用)。"""
        total = 0
        for source in filters.sources:
            cfg = _SOURCES[source]
            row = db.execute(
                text(
                    f"SELECT COUNT(*) AS n FROM {cfg['parent']} AS trans "
                    f"WHERE trans.transcript IS NOT NULL AND trans.transcript != ''"
                    f"{_DATE_WHERE}"
                ),
                filters.date_params(),
            ).mappings().first()
            total += int(row["n"] if row else 0)
        return total

    def browse_recent(
        self,
        db: Session,
        filters: SearchFilters,
        max_recordings: int,
    ) -> List[Dict]:
        """期間内の録音を新しい順に返す(要約・履歴参照用)。

        チャンクの有無に依存せず親テーブルから直接選ぶため、
        埋め込み未作成の録音も履歴参照では確実に見える。
        """
        rows: List[Dict] = []
        for source in filters.sources:
            cfg = _SOURCES[source]
            stmt = text(
                f"SELECT trans.id AS transcription_id, {cfg['title']} AS title, "
                f"trans.tags AS tags, trans.recorded_date AS recorded_date, "
                f"trans.created_at AS recorded_at, trans.duration_seconds AS duration, "
                f"'{source}' AS source "
                f"FROM {cfg['parent']} AS trans "
                f"WHERE trans.transcript IS NOT NULL AND trans.transcript != ''{_DATE_WHERE} "
                f"ORDER BY trans.recorded_date DESC, trans.id DESC LIMIT :max_recs"
            )
            params = {"max_recs": max_recordings, **filters.date_params()}
            for r in db.execute(stmt, params).mappings().all():
                rec = dict(r)
                rec["chunk_id"] = None
                rec["chunk_index"] = None
                rec["chunk_text"] = None
                rec["score"] = 1.0
                rec["score_vector"] = None
                rec["score_fts"] = None
                rows.append(rec)
        rows.sort(key=lambda r: (r.get("recorded_date") or "", r.get("transcription_id") or 0), reverse=True)
        return rows[:max_recordings]

    def corpus_stats(self, db: Session) -> Dict[str, Dict]:
        """UI表示用のコーパス統計。"""
        stats: Dict[str, Dict] = {}
        for source, cfg in _SOURCES.items():
            row = db.execute(
                text(
                    f"SELECT COUNT(*) AS n, MAX(recorded_date) AS latest "
                    f"FROM {cfg['parent']} WHERE transcript IS NOT NULL AND transcript != ''"
                )
            ).mappings().first()
            stats[source] = {"count": row["n"] if row else 0, "latest": row["latest"] if row else None}
        return stats

    def available_date_range(self, db: Session, sources: Sequence[str]) -> Optional[Tuple[str, str]]:
        """データが存在する日付範囲(該当なし時の案内用)。"""
        lo: Optional[str] = None
        hi: Optional[str] = None
        for source in sources:
            cfg = _SOURCES[source]
            row = db.execute(
                text(f"SELECT MIN(recorded_date) AS lo, MAX(recorded_date) AS hi FROM {cfg['parent']}")
            ).mappings().first()
            if row and row["lo"]:
                lo = min(lo, row["lo"]) if lo else row["lo"]
                hi = max(hi, row["hi"]) if hi else row["hi"]
        return (lo, hi) if lo and hi else None
