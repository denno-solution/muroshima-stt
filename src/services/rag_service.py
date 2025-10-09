"""RAG向けの埋め込み生成と保存ロジック。PostgresとTurso双方に対応。"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, Iterable, List, Tuple

from openai import OpenAI
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from models import (
    AudioTranscription,
    AudioTranscriptionChunk,
    EMBEDDING_DIM,
    LIBSQL_VECTOR_INDEX_NAME,
    USE_VECTOR,
    VECTOR_BACKEND,
)

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "600"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
COMPLETION_MODEL = os.getenv("RAG_COMPLETION_MODEL", "gpt-4o-mini")
ENABLE_RAG = os.getenv("ENABLE_RAG", "true").lower() in {"1", "true", "yes", "on"}

# Hybrid search parameters
HYBRID_DEFAULT_ALPHA = float(os.getenv("RAG_HYBRID_ALPHA", "0.6"))  # ベクトル寄り
HYBRID_CAND_MULT = int(os.getenv("RAG_HYBRID_CAND_MULT", "3"))  # 候補母集団の拡大係数
ENABLE_FTS = os.getenv("ENABLE_FTS", "true").lower() in {"1", "true", "yes", "on"}

# Prompt safety limits（環境変数で調整可能）
CONTEXT_MAX_CHUNKS = int(os.getenv("RAG_CONTEXT_MAX_CHUNKS", "12"))
CONTEXT_MAX_CHARS = int(os.getenv("RAG_CONTEXT_MAX_CHARS", "20000"))  # おおよそ数千トークン相当

# Retrieval breadth（検索候補の母集団サイズ）。インデックスは常に全体を対象に上位を返します。
RETRIEVAL_K = int(os.getenv("RAG_RETRIEVAL_K", "100"))


class RAGService:
    """埋め込み管理と検索ロジック。pgvector/libSQL双方で動作。"""

    def __init__(self) -> None:
        self._enabled = bool(USE_VECTOR) and ENABLE_RAG
        self._vector_backend = VECTOR_BACKEND
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            if self._enabled:
                logger.warning("OPENAI_API_KEY が未設定のため RAG を無効化します")
            self._enabled = False
        self._client = OpenAI() if self._enabled else None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def index_transcription(self, db: Session, transcription_id: int, text: str) -> None:
        """文字起こし全文をチャンク化して埋め込みを保存。"""

        if not self.enabled:
            return

        chunks = list(self._chunk_text(text))
        if not chunks:
            logger.debug("RAG: チャンクなしのためスキップ (transcription_id=%s)", transcription_id)
            return

        embeddings = self._embed_texts(chunks)
        if not embeddings:
            logger.warning("RAG: 埋め込み生成に失敗したためスキップ (transcription_id=%s)", transcription_id)
            return

        # 既存チャンクを削除してから再作成
        db.query(AudioTranscriptionChunk).filter_by(transcription_id=transcription_id).delete()

        for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            chunk = AudioTranscriptionChunk(
                transcription_id=transcription_id,
                chunk_index=idx,
                chunk_text=chunk_text,
                embedding=embedding,
            )
            db.add(chunk)

    def similarity_search(self, db: Session, query: str, top_k: int = 5) -> List[Dict]:
        if not self.enabled:
            return []

        query_embedding = self._embed_texts([query])
        if not query_embedding:
            return []

        query_vector = query_embedding[0]

        if self._vector_backend == "postgres":
            distance_expr = AudioTranscriptionChunk.embedding.cosine_distance(query_vector)

            stmt = (
                select(
                    AudioTranscriptionChunk.id.label("chunk_id"),
                    AudioTranscriptionChunk.chunk_text.label("chunk_text"),
                    AudioTranscriptionChunk.chunk_index.label("chunk_index"),
                    AudioTranscription.音声ID.label("transcription_id"),
                    AudioTranscription.音声ファイルpath.label("file_path"),
                    AudioTranscription.タグ.label("tag"),
                    AudioTranscription.録音時刻.label("recorded_at"),
                    AudioTranscription.録音時間.label("duration"),
                    distance_expr.label("distance"),
                )
                .select_from(AudioTranscriptionChunk)
                .join(
                    AudioTranscription,
                    AudioTranscription.音声ID == AudioTranscriptionChunk.transcription_id,
                )
                .order_by(distance_expr)
                .limit(top_k)
            )

            rows = db.execute(stmt).mappings().all()
        elif self._vector_backend == "libsql":
            rows = self._similarity_search_libsql(db, query_vector, top_k)
        else:
            return []

        matches: List[Dict] = []
        for row in rows:
            distance = row["distance"] or 0.0
            score = max(0.0, 1.0 - float(distance))
            matches.append(
                {
                    "chunk_id": row["chunk_id"],
                    "chunk_text": row["chunk_text"],
                    "chunk_index": row["chunk_index"],
                    "transcription_id": row["transcription_id"],
                    "file_path": row["file_path"],
                    "tag": row.get("tag"),
                    "recorded_at": row.get("recorded_at"),
                    "duration": row.get("duration"),
                    "distance": float(distance),
                    "score": score,
                }
            )

        return matches

    def similarity_search_hybrid(
        self,
        db: Session,
        query: str,
        top_k: int = 5,
        alpha: float = HYBRID_DEFAULT_ALPHA,
    ) -> List[Dict]:
        """FTS × ベクトルのハイブリッド検索。

        alpha: 0.0〜1.0（1.0でベクトルのみ、0.0でFTSのみ）
        """
        if not self.enabled or not ENABLE_FTS:
            return self.similarity_search(db, query, top_k)

        # 埋め込み生成
        qvecs = self._embed_texts([query])
        if not qvecs:
            # ベクトルが使えない場合はFTSのみ
            return self._similarity_search_hybrid_fts_only(db, query, top_k)
        qvec = qvecs[0]

        # 候補母集団の件数
        cand_k = max(top_k * HYBRID_CAND_MULT, top_k)

        if self._vector_backend == "postgres":
            return self._hybrid_postgres(db, query, qvec, top_k, cand_k, alpha)
        elif self._vector_backend == "libsql":
            return self._hybrid_libsql(db, query, qvec, top_k, cand_k, alpha)
        else:
            return []

    # --- libSQL (Turso) 実装 ---
    def _hybrid_libsql(
        self,
        db: Session,
        query: str,
        query_vector: List[float],
        top_k: int,
        cand_k: int,
        alpha: float,
    ) -> List[Dict]:
        # ベクトル候補
        vec_rows = self._libsql_vector_candidates(db, query_vector, cand_k)
        # FTS候補
        fts_rows = self._libsql_fts_candidates(db, query, cand_k)

        return self._blend_and_fetch_libsql(db, vec_rows, fts_rows, top_k, alpha)

    def _libsql_vector_candidates(self, db: Session, qvec: List[float], k: int) -> List[Dict]:
        stmt = text(
            "SELECT id, distance FROM vector_top_k(:index_name, vector32(:q), :k)"
        )
        rows = db.execute(
            stmt,
            {"index_name": LIBSQL_VECTOR_INDEX_NAME, "q": json.dumps(qvec), "k": k},
        ).mappings().all()
        return [dict(row) for row in rows]

    def _libsql_fts_candidates(self, db: Session, query: str, k: int) -> List[Dict]:
        # FTS5: bm25は小さいほど良い。後で 1/(1+bm25) に変換
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
            # FTS未構成時のフォールバック（LIKE検索、スコアは一律0.5）
            like_stmt = text(
                "SELECT id, 0.5 AS like_score FROM audio_transcription_chunks WHERE chunk_text LIKE :pat LIMIT :k"
            )
            rows = db.execute(
                like_stmt, {"pat": f"%{query}%", "k": k}
            ).mappings().all()
            # 互換のため bm25 に変換（0.5 -> s_fts=0.5 => bm25 ~1.0 とみなす）
            rows = [
                {"id": r["id"], "bm25": 1.0}
                for r in rows
            ]
        return [dict(row) for row in rows]

    def _blend_and_fetch_libsql(
        self,
        db: Session,
        vec_rows: List[Dict],
        fts_rows: List[Dict],
        top_k: int,
        alpha: float,
    ) -> List[Dict]:
        # 正規化と結合
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

        # メタ情報取得
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

    def _similarity_search_hybrid_fts_only(self, db: Session, query: str, top_k: int) -> List[Dict]:
        # ベクトルが使えない場合の単純FTS検索（libSQL想定）
        rows = self._libsql_fts_candidates(db, query, top_k)
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

    # --- Postgres 実装（簡易版） ---
    def _hybrid_postgres(
        self,
        db: Session,
        query: str,
        query_vector: List[float],
        top_k: int,
        cand_k: int,
        alpha: float,
    ) -> List[Dict]:
        # ベクトル候補
        distance_expr = AudioTranscriptionChunk.embedding.cosine_distance(query_vector)
        vec_stmt = (
            select(
                AudioTranscriptionChunk.id.label("id"),
                distance_expr.label("distance"),
            )
            .select_from(AudioTranscriptionChunk)
            .order_by(distance_expr)
            .limit(cand_k)
        )
        vec_rows = db.execute(vec_stmt).mappings().all()

        # FTS候補（simple辞書、ts_rank_cdでスコア取得）
        fts_stmt = text(
            """
            SELECT id, ts_rank_cd(to_tsvector('simple', chunk_text), plainto_tsquery('simple', :q)) AS rank
            FROM audio_transcription_chunks
            WHERE to_tsvector('simple', chunk_text) @@ plainto_tsquery('simple', :q)
            ORDER BY rank DESC
            LIMIT :k
            """
        )
        fts_rows = db.execute(fts_stmt, {"q": query, "k": cand_k}).mappings().all()

        # 正規化・結合
        vec_map = {int(r["id"]): float(1.0 - float(r["distance"])) for r in vec_rows}
        # ts_rank_cd は 0〜1 程度の値が返る想定
        fts_map = {int(r["id"]): float(max(0.0, float(r["rank"]))) for r in fts_rows}

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

    def answer(
        self,
        db: Session,
        query: str,
        top_k: int | None = None,
        hybrid: bool = False,
        alpha: float = HYBRID_DEFAULT_ALPHA,
        context_k: int | None = None,
    ) -> Dict:
        # 取得する候補件数（検索母集団からの上位件数）。UIには出さない。
        tk = int(top_k or RETRIEVAL_K)
        matches_all = (
            self.similarity_search_hybrid(db, query, tk, alpha)
            if hybrid
            else self.similarity_search(db, query, tk)
        )
        if not matches_all:
            return {"answer": "関連するテキストが見つかりませんでした。", "matches": []}

        # プロンプト用に上限を適用（件数/文字数）
        use_k = int(context_k or CONTEXT_MAX_CHUNKS)
        selected: List[Dict] = []
        used_chars = 0
        for m in matches_all:
            if len(selected) >= use_k:
                break
            txt = m.get("chunk_text") or ""
            # ヘッダ分のメタ情報余白も少し見込む（+128）
            add_len = len(txt) + 128
            if used_chars + add_len > CONTEXT_MAX_CHARS:
                break
            selected.append(m)
            used_chars += add_len

        if not selected:
            # どれも長すぎて入らない場合は最上位1件だけトリムして使用
            head = matches_all[0]
            trimmed = dict(head)
            trimmed["chunk_text"] = (head.get("chunk_text") or "")[: max(200, CONTEXT_MAX_CHARS // 2)]
            selected = [trimmed]

        prompt = self._build_prompt(query, selected)
        answer = self._generate_answer(prompt)

        return {
            "answer": answer,
            "matches": selected,
            "meta": {
                "candidates": len(matches_all),
                "used_context_chunks": len(selected),
                "used_context_chars": used_chars,
                "limits": {
                    "max_chunks": use_k,
                    "max_chars": CONTEXT_MAX_CHARS,
                },
            },
        }

    def _similarity_search_libsql(
        self, db: Session, query_vector: List[float], top_k: int
    ) -> List[Dict]:
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
                matches.distance AS distance
            FROM vector_top_k(:index_name, vector32(:query_vector), :top_k) AS matches
            JOIN audio_transcription_chunks AS chunk ON chunk.id = matches.id
            JOIN audio_transcriptions AS trans ON trans."音声ID" = chunk.transcription_id
            ORDER BY matches.distance ASC
            """
        )

        params = {
            "index_name": LIBSQL_VECTOR_INDEX_NAME,
            "query_vector": vector_literal,
            "top_k": top_k,
        }
        rows = db.execute(stmt, params).mappings().all()
        return rows

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not self._client:
            return []
        try:
            response = self._client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        except Exception as exc:  # pragma: no cover - APIエラー
            logger.error("OpenAI embeddings API 呼び出しで失敗: %s", exc)
            return []

        embeddings: List[List[float]] = []
        for item in response.data:
            embedding = getattr(item, "embedding", None)
            if embedding and len(embedding) == EMBEDDING_DIM:
                embeddings.append(list(embedding))
            else:
                logger.warning(
                    "RAG: 埋め込みベクトルの次元が想定と異なります (expected=%s, actual=%s)",
                    EMBEDDING_DIM,
                    len(embedding) if embedding else None,
                )
        return embeddings

    def _chunk_text(self, text: str) -> Iterable[str]:
        """句点ベースでチャンク化。日本語/英語混在にも対応するための簡易実装。"""

        if not text:
            return []

        sentences = [s.strip() for s in re.split(r"(?<=[。．.!?！？])", text) if s and s.strip()]
        if not sentences:
            sentences = [text.strip()]

        chunks: List[str] = []
        current: List[str] = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_length = len(sentence)
            if current_length + sentence_length <= DEFAULT_CHUNK_SIZE:
                current.append(sentence)
                current_length += sentence_length
                continue

            # flush current chunk
            if current:
                chunks.append("".join(current))

            # overlap処理
            if DEFAULT_CHUNK_OVERLAP > 0 and chunks:
                overlap_text = chunks[-1][-DEFAULT_CHUNK_OVERLAP:]
                current = [overlap_text, sentence]
                current_length = len(overlap_text) + sentence_length
            else:
                current = [sentence]
                current_length = sentence_length

        if current:
            chunks.append("".join(current))

        return chunks

    def _build_prompt(self, query: str, matches: List[Dict]) -> str:
        context_lines = []
        for match in matches:
            meta_parts = []
            if match.get("file_path"):
                meta_parts.append(f"ファイル: {match['file_path']}")
            if match.get("tag"):
                meta_parts.append(f"タグ: {match['tag']}")
            if match.get("recorded_at"):
                meta_parts.append(f"録音時刻: {match['recorded_at']}")
            meta = " / ".join(meta_parts)
            header = f"[スコア: {match['score']:.3f}] {meta}" if meta else f"[スコア: {match['score']:.3f}]"
            context_lines.append(f"{header}\n{match['chunk_text']}")

        context_block = "\n\n".join(context_lines)

        instructions = (
            "あなたは社内の音声文字起こしデータから質問に答えるアシスタントです。"
            "以下のコンテキストのみを根拠に、根拠が無ければその旨を明示して日本語で簡潔に回答してください。"
        )
        return f"{instructions}\n\nコンテキスト:\n{context_block}\n\n質問:\n{query}"

    def _generate_answer(self, prompt: str) -> str:
        if not self._client:
            return ""

        try:
            response = self._client.chat.completions.create(
                model=COMPLETION_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "あなたは音声文字起こしデータを根拠に回答する日本語アシスタントです。根拠が無い時はその旨を伝えてください。",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
        except Exception as exc:  # pragma: no cover - APIエラー
            msg = str(exc)
            logger.error("OpenAI chat_completions API 呼び出しで失敗: %s", msg)
            if "maximum context length" in msg or "context_length_exceeded" in msg or "too many tokens" in msg:
                return (
                    "回答生成時にプロンプトが長過ぎました。検索上限または 'RAG_CONTEXT_MAX_*' を下げて再実行してください。"
                )
            return "回答生成中にエラーが発生しました。ログを確認してください。"

        choice = response.choices[0] if response.choices else None
        if not choice or not choice.message or not choice.message.content:
            return "回答を生成できませんでした。"
        return choice.message.content.strip()


rag_service = RAGService()


def get_rag_service() -> RAGService:
    return rag_service
