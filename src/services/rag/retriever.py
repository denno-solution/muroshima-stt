from __future__ import annotations

import json
from typing import Dict, List, Tuple

from sqlalchemy import text
from sqlalchemy.orm import Session

from models import LIBSQL_VECTOR_INDEX_NAME


class LibsqlRetriever:
    """libSQL向けのベクトル/FTS検索ヘルパー。"""

    def __init__(self, index_name: str = LIBSQL_VECTOR_INDEX_NAME) -> None:
        self.index_name = index_name

    # --- 公開メソッド ---
    def similarity_search(self, db: Session, query_vector: List[float], top_k: int) -> List[Dict]:
        vector_literal = json.dumps(query_vector)
        stmt = text(
            """
            SELECT
                chunk.id AS chunk_id,
                chunk.chunk_text AS chunk_text,
                chunk.chunk_index AS chunk_index,
                trans."音声ID" AS transcription_id,
                trans."音声ファイルpath" AS file_path,
                trans."タグ" AS tag,
                trans."録音時刻" AS recorded_at,
                trans."録音時間" AS duration,
                vector_distance_cos(chunk.embedding, vector32(:query_vector)) AS distance
            FROM vector_top_k(:index_name, vector32(:query_vector), :top_k) AS matches
            JOIN audio_transcription_chunks AS chunk ON chunk.id = matches.id
            JOIN audio_transcriptions AS trans ON trans."音声ID" = chunk.transcription_id
            ORDER BY distance ASC
            """
        )

        params = {
            "index_name": self.index_name,
            "query_vector": vector_literal,
            "top_k": top_k,
        }
        rows = db.execute(stmt, params).mappings().all()
        return [dict(r) for r in rows]

    def hybrid_search(
        self,
        db: Session,
        query: str,
        query_vector: List[float],
        top_k: int,
        cand_k: int,
        alpha: float,
    ) -> List[Dict]:
        vec_rows = self._vector_candidates(db, query_vector, cand_k)
        fts_rows = self._fts_candidates(db, query, cand_k)
        return self._blend_and_fetch(db, vec_rows, fts_rows, top_k, alpha)

    def fts_only(self, db: Session, query: str, top_k: int) -> List[Dict]:
        rows = self._fts_candidates(db, query, top_k)
        ids = [int(r["id"]) for r in rows]
        if not ids:
            return []
        ids_sql = ",".join(str(int(i)) for i in ids)
        meta = db.execute(
            text(
                f"""
                SELECT
                    chunk.id AS chunk_id,
                    chunk.chunk_text AS chunk_text,
                    chunk.chunk_index AS chunk_index,
                    trans."音声ID" AS transcription_id,
                    trans."音声ファイルpath" AS file_path,
                    trans."タグ" AS tag,
                    trans."録音時刻" AS recorded_at,
                    trans."録音時間" AS duration
                FROM audio_transcription_chunks AS chunk
                JOIN audio_transcriptions AS trans ON trans."音声ID" = chunk.transcription_id
                WHERE chunk.id IN ({ids_sql})
                """
            )
        ).mappings().all()

        row_map = {int(r["chunk_id"]): r for r in meta}
        matches: List[Dict] = []
        for r in rows[:top_k]:
            base = row_map.get(int(r["id"]))
            if not base:
                continue
            sim_fts = 1.0 / (1.0 + max(0.0, float(r.get("bm25", 1.0))))
            rec = dict(base)
            rec["score"] = float(sim_fts)
            rec["score_vector"] = 0.0
            rec["score_fts"] = float(sim_fts)
            matches.append(rec)
        return matches

    # --- 内部ヘルパー ---
    def _vector_candidates(self, db: Session, qvec: List[float], k: int) -> List[Dict]:
        stmt = text(
            """
            SELECT
                i.id AS id,
                vector_distance_cos(chunk.embedding, vector32(:q)) AS distance
            FROM vector_top_k(:index_name, vector32(:q), :k) AS i
            JOIN audio_transcription_chunks AS chunk ON chunk.id = i.id
            """
        )
        rows = db.execute(
            stmt,
            {"index_name": self.index_name, "q": json.dumps(qvec), "k": k},
        ).mappings().all()
        return [dict(row) for row in rows]

    def _fts_candidates(self, db: Session, query: str, k: int) -> List[Dict]:
        try:
            stmt = text(
                """
                SELECT rowid AS id, bm25(audio_transcription_chunks_fts) AS bm25
                FROM audio_transcription_chunks_fts
                WHERE audio_transcription_chunks_fts MATCH :q
                ORDER BY bm25 LIMIT :k
                """
            )
            rows = db.execute(stmt, {"q": query, "k": k}).mappings().all()
        except Exception:
            like_stmt = text(
                "SELECT id, 0.5 AS like_score FROM audio_transcription_chunks WHERE chunk_text LIKE :pat LIMIT :k"
            )
            rows = db.execute(like_stmt, {"pat": f"%{query}%", "k": k}).mappings().all()
            rows = [{"id": r["id"], "bm25": 1.0} for r in rows]
        return [dict(r) for r in rows]

    def _blend_and_fetch(
        self,
        db: Session,
        vec_rows: List[Dict],
        fts_rows: List[Dict],
        top_k: int,
        alpha: float,
    ) -> List[Dict]:
        vec_map = {int(r["id"]): float(1.0 - float(r["distance"])) for r in vec_rows}
        fts_map = {int(r["id"]): float(1.0 / (1.0 + max(0.0, float(r["bm25"])))) for r in fts_rows}

        ids = set(vec_map) | set(fts_map)
        scored: List[Tuple[int, float, float, float]] = []
        for cid in ids:
            v = max(0.0, min(1.0, vec_map.get(cid, 0.0)))
            f = max(0.0, min(1.0, fts_map.get(cid, 0.0)))
            s = alpha * v + (1.0 - alpha) * f
            scored.append((cid, s, v, f))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_ids = [cid for cid, _, _, _ in scored[:top_k]]
        if not top_ids:
            return []

        ids_sql = ",".join(str(int(i)) for i in top_ids)
        rows = db.execute(
            text(
                f"""
                SELECT
                    chunk.id AS chunk_id,
                    chunk.chunk_text AS chunk_text,
                    chunk.chunk_index AS chunk_index,
                    trans."音声ID" AS transcription_id,
                    trans."音声ファイルpath" AS file_path,
                    trans."タグ" AS tag,
                    trans."録音時刻" AS recorded_at,
                    trans."録音時間" AS duration
                FROM audio_transcription_chunks AS chunk
                JOIN audio_transcriptions AS trans ON trans."音声ID" = chunk.transcription_id
                WHERE chunk.id IN ({ids_sql})
                """
            )
        ).mappings().all()

        row_map = {int(r["chunk_id"]): r for r in rows}
        matches: List[Dict] = []
        for cid, s, v, f in scored[:top_k]:
            base = row_map.get(int(cid))
            if not base:
                continue
            rec = dict(base)
            rec["score"] = float(s)
            rec["score_vector"] = float(v)
            rec["score_fts"] = float(f)
            matches.append(rec)
        return matches
