"""RAG向けの埋め込み生成と保存ロジック。Turso(libSQL)専用。"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, date, timedelta
from typing import Dict, Iterable, List, Tuple, Optional

from openai import OpenAI
from sqlalchemy import text
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
# 2025-10 時点：gpt-5 系列を既定に（Responses API 対応、品質/コスパ良好）。
COMPLETION_MODEL = os.getenv("RAG_COMPLETION_MODEL", "gpt-5-mini")
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


def parse_date_from_query(query: str) -> Optional[Tuple[date, date]]:
    """クエリから日付範囲を抽出する。

    対応パターン:
    - 「12月3日」「12/3」「12-3」 → 今年のその日付
    - 「2024年12月3日」「2024/12/3」 → 指定年月日
    - 「先月」「今月」「先週」「今週」「昨日」「今日」「一昨日」
    - 「◯日前」「◯週間前」「◯ヶ月前」

    Returns:
        (start_date, end_date) のタプル、または None
    """
    today = date.today()
    current_year = today.year

    # 相対日付パターン
    if "今日" in query:
        return (today, today)
    if "昨日" in query:
        yesterday = today - timedelta(days=1)
        return (yesterday, yesterday)
    if "一昨日" in query or "おととい" in query:
        day_before = today - timedelta(days=2)
        return (day_before, day_before)
    if "今週" in query:
        start = today - timedelta(days=today.weekday())
        end = start + timedelta(days=6)
        return (start, min(end, today))
    if "先週" in query:
        start = today - timedelta(days=today.weekday() + 7)
        end = start + timedelta(days=6)
        return (start, end)
    if "今月" in query:
        start = today.replace(day=1)
        return (start, today)
    if "先月" in query:
        first_of_month = today.replace(day=1)
        last_month_end = first_of_month - timedelta(days=1)
        last_month_start = last_month_end.replace(day=1)
        return (last_month_start, last_month_end)

    # 「◯日前」「◯週間前」「◯ヶ月前」パターン
    days_ago = re.search(r"(\d+)\s*日前", query)
    if days_ago:
        n = int(days_ago.group(1))
        target = today - timedelta(days=n)
        return (target, target)

    weeks_ago = re.search(r"(\d+)\s*週間?前", query)
    if weeks_ago:
        n = int(weeks_ago.group(1))
        target_end = today - timedelta(weeks=n)
        target_start = target_end - timedelta(days=6)
        return (target_start, target_end)

    months_ago = re.search(r"(\d+)\s*[ヶか]?月前", query)
    if months_ago:
        n = int(months_ago.group(1))
        # 簡易的に30日単位で計算
        target = today - timedelta(days=30 * n)
        month_start = target.replace(day=1)
        if target.month == 12:
            month_end = target.replace(day=31)
        else:
            month_end = target.replace(month=target.month + 1, day=1) - timedelta(days=1)
        return (month_start, month_end)

    # 具体的な日付パターン: 「2024年12月3日」「2024/12/3」「2024-12-3」
    full_date = re.search(r"(\d{4})[年/\-](\d{1,2})[月/\-](\d{1,2})日?", query)
    if full_date:
        try:
            year = int(full_date.group(1))
            month = int(full_date.group(2))
            day = int(full_date.group(3))
            target = date(year, month, day)
            return (target, target)
        except ValueError:
            pass

    # 月日パターン: 「12月3日」「12/3」「12-3」
    month_day = re.search(r"(\d{1,2})[月/\-](\d{1,2})日?", query)
    if month_day:
        try:
            month = int(month_day.group(1))
            day = int(month_day.group(2))
            # 今年のその日付を試す。未来なら去年
            target = date(current_year, month, day)
            if target > today:
                target = date(current_year - 1, month, day)
            return (target, target)
        except ValueError:
            pass

    return None


def highlight_date_in_query(query: str) -> str:
    """クエリ内の日付パターンをStreamlitのカラーマークダウンでハイライトする。"""
    result = query

    # ハイライト用のラッパー関数
    def wrap(match: re.Match) -> str:
        return f":orange[{match.group(0)}]"

    # 相対日付キーワード
    relative_patterns = [
        r"今日",
        r"昨日",
        r"一昨日",
        r"おととい",
        r"今週",
        r"先週",
        r"今月",
        r"先月",
        r"\d+\s*日前",
        r"\d+\s*週間?前",
        r"\d+\s*[ヶか]?月前",
    ]

    for pattern in relative_patterns:
        result = re.sub(pattern, wrap, result)

    # 具体的な日付パターン（年月日）
    result = re.sub(r"(\d{4})[年/\-](\d{1,2})[月/\-](\d{1,2})日?", wrap, result)

    # 月日パターン
    result = re.sub(r"(\d{1,2})[月/\-](\d{1,2})日?", wrap, result)

    return result


class RAGService:
    """埋め込み管理と検索ロジック。Turso(libSQL)専用で動作。"""

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

        if self._vector_backend == "libsql":
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

        if self._vector_backend == "libsql":
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
        # libSQL の vector_top_k は距離を返さないため、基表に JOIN して
        # vector_distance_cos で距離を計算してから返す。
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

    # Postgres実装は削除（Turso専用化）

    # 一括生成APIは廃止（streaming専用化）

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
                vector_distance_cos(chunk.embedding, vector32(:query_vector)) AS distance
            FROM vector_top_k(:index_name, vector32(:query_vector), :top_k) AS matches
            JOIN audio_transcription_chunks AS chunk ON chunk.id = matches.id
            JOIN audio_transcriptions AS trans ON trans."音声ID" = chunk.transcription_id
            ORDER BY distance ASC
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
        """回答用のプロンプトを生成。コンテキストに通し番号を付与し、引用しやすくする。"""
        numbered_context = []
        for i, match in enumerate(matches, start=1):
            meta_parts = []
            if match.get("file_path"):
                meta_parts.append(f"ファイル: {match['file_path']}")
            if match.get("tag"):
                meta_parts.append(f"タグ: {match['tag']}")
            if match.get("recorded_at"):
                meta_parts.append(f"録音時刻: {match['recorded_at']}")
            meta = " / ".join(meta_parts)
            header = (
                f"[#{i} スコア:{match['score']:.3f}] {meta}" if meta else f"[#{i} スコア:{match['score']:.3f}]"
            )
            numbered_context.append(f"{header}\n{match['chunk_text']}")

        context_block = "\n\n".join(numbered_context)

        # 出力スタイルはここで明示する（拒否よりも「分かっていること/不足していること」を優先）。
        instructions = (
            "あなたは社内の音声文字起こしデータを根拠に回答する日本語アシスタントです。"  # 役割
            "事実は必ず下のコンテキスト内から根拠を取り、出典として [#番号] を示してください。"  # 根拠と引用
            "根拠が完全には揃わない場合でも、\"分かっていること\"と\"不足情報\"を分けて簡潔に答えてください。"  # 過度な拒否の抑制
            "日付や時刻は可能なら YYYY-MM-DD 形式で明示してください。"  # 日付の明確化
        )

        # 回答のフォーマットを固定化して安定させる
        output_format = (
            "出力は次の3セクションで返してください:\n"
            "1) 回答:\n- 箇条書きで要点のみ（最大5項目）。\n"
            "2) 根拠:\n- 参照した [#番号] と短い引用/要約（1〜3件）。\n"
            "3) 不足情報/前提:\n- 追加で必要な情報や不確実な点。"
        )

        return (
            f"{instructions}\n\n"
            f"コンテキスト（番号付き）:\n{context_block}\n\n"
            f"質問:\n{query}\n\n"
            f"{output_format}"
        )

    def _generate_answer(self, prompt: str) -> str:
        if not self._client:
            return ""

        try:
            # Responses API を使用（messages ではなく input）。温度は未指定＝既定値。
            response = self._client.responses.create(
                model=COMPLETION_MODEL,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "あなたはRAGベースの社内QAアシスタントです。"
                            "事実は必ず与えられたコンテキストに基づき、出典として [#番号] を明記してください。"
                            "コンテキスト外の推測はしないでください。足りない点は『不足情報』に列挙します。"
                            "文体は簡潔で日本語、箇条書きを優先します。"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception as exc:  # pragma: no cover - APIエラー
            msg = str(exc)
            logger.error("OpenAI Responses API 呼び出しで失敗: %s", msg)
            if "maximum context length" in msg or "context_length_exceeded" in msg or "too many tokens" in msg:
                return (
                    "回答生成時にプロンプトが長過ぎました。検索上限または 'RAG_CONTEXT_MAX_*' を下げて再実行してください。"
                )
            return "回答生成中にエラーが発生しました。ログを確認してください。"

        # SDK の output_text ヘルパーを優先使用（無い場合はフォールバック）
        try:
            text_out = getattr(response, "output_text", None)
            if isinstance(text_out, str) and text_out.strip():
                return text_out.strip()
        except Exception:
            pass

        # フォールバック：output.items からテキストを連結（Responses API仕様の将来変化に備える）
        out = []
        try:
            for item in getattr(response, "output", []) or []:
                # item には {type: "message", content: [{type:"text", text:"..."}, ...]} 等が入る想定
                contents = getattr(item, "content", None) or item.get("content") if isinstance(item, dict) else None
                if isinstance(contents, list):
                    for c in contents:
                        t = getattr(c, "text", None) or (c.get("text") if isinstance(c, dict) else None)
                        if isinstance(t, str):
                            out.append(t)
        except Exception:
            pass

        text_joined = "\n".join(out).strip()
        return text_joined or "回答を生成できませんでした。"

    def _filter_by_date(self, matches: List[Dict], date_range: Tuple[date, date]) -> List[Dict]:
        """検索結果を日付範囲でフィルタリング。"""
        start_date, end_date = date_range
        filtered = []
        for m in matches:
            recorded_at = m.get("recorded_at")
            if not recorded_at:
                continue
            # datetime型またはstring型を処理
            if isinstance(recorded_at, str):
                try:
                    recorded_date = datetime.fromisoformat(recorded_at.replace("Z", "+00:00")).date()
                except (ValueError, TypeError):
                    try:
                        # 別のフォーマットを試す
                        recorded_date = datetime.strptime(recorded_at[:10], "%Y-%m-%d").date()
                    except (ValueError, TypeError):
                        continue
            elif isinstance(recorded_at, datetime):
                recorded_date = recorded_at.date()
            elif isinstance(recorded_at, date):
                recorded_date = recorded_at
            else:
                continue

            if start_date <= recorded_date <= end_date:
                filtered.append(m)
        return filtered

    def _build_chat_prompt(
        self,
        query: str,
        matches: List[Dict],
        chat_history: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """会話履歴を含むプロンプトを生成。OpenAI Responses API形式で返す。"""
        # コンテキストブロックを生成
        numbered_context = []
        for i, match in enumerate(matches, start=1):
            meta_parts = []
            if match.get("file_path"):
                meta_parts.append(f"ファイル: {match['file_path']}")
            if match.get("tag"):
                meta_parts.append(f"タグ: {match['tag']}")
            if match.get("recorded_at"):
                recorded = match["recorded_at"]
                if isinstance(recorded, datetime):
                    recorded = recorded.strftime("%Y-%m-%d %H:%M")
                elif isinstance(recorded, date):
                    recorded = recorded.strftime("%Y-%m-%d")
                meta_parts.append(f"録音日時: {recorded}")
            meta = " / ".join(meta_parts)
            header = (
                f"[#{i} スコア:{match['score']:.3f}] {meta}" if meta else f"[#{i} スコア:{match['score']:.3f}]"
            )
            numbered_context.append(f"{header}\n{match['chunk_text']}")

        context_block = "\n\n".join(numbered_context)

        system_content = (
            "あなたはRAGベースの社内QAアシスタントです。"
            "事実は必ず与えられたコンテキストに基づき、出典として [#番号] を明記してください。"
            "コンテキスト外の推測はしないでください。足りない点は『不足情報』に列挙します。"
            "文体は簡潔で日本語、箇条書きを優先します。"
            "会話の文脈を維持し、前の質問への回答と関連付けて答えてください。"
        )

        messages = [{"role": "system", "content": system_content}]

        # 会話履歴を追加（最新5ターン程度に制限）
        if chat_history:
            # 履歴は (user, assistant) のペアで構成されている想定
            recent_history = chat_history[-10:]  # 最新10メッセージ
            for msg in recent_history:
                role = msg.get("role")
                content = msg.get("content", "")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})

        # 現在のクエリとコンテキストを追加
        user_prompt = (
            f"以下のコンテキスト（番号付き）を参照して質問に答えてください。\n\n"
            f"コンテキスト:\n{context_block}\n\n"
            f"質問:\n{query}\n\n"
            f"出力は次の3セクションで返してください:\n"
            f"1) 回答: 箇条書きで要点のみ（最大5項目）。\n"
            f"2) 根拠: 参照した [#番号] と短い引用/要約（1〜3件）。\n"
            f"3) 不足情報/前提: 追加で必要な情報や不確実な点。"
        )
        messages.append({"role": "user", "content": user_prompt})

        return messages

    # --- Streaming API ---
    def answer_stream(
        self,
        db: Session,
        query: str,
        top_k: int | None = None,
        hybrid: bool = False,
        alpha: float = HYBRID_DEFAULT_ALPHA,
        context_k: int | None = None,
        chat_history: Optional[List[Dict]] = None,
    ) -> Dict:
        """検索→プロンプト生成までを先に実行し、テキスト生成はストリーミングで返す。

        Args:
            chat_history: 過去の会話履歴 [{"role": "user"|"assistant", "content": "..."}, ...]

        戻り値に `stream_fn`（呼び出すとジェネレータを返す関数）を含める。
        UI側で `st.write_stream(result['stream_fn']())` などで逐次表示できる。
        """
        if not self.enabled or not self._client:
            return {"matches": [], "meta": {}, "stream_fn": lambda: iter(())}

        t0 = time.time()
        tk = int(top_k or RETRIEVAL_K)
        matches_all = (
            self.similarity_search_hybrid(db, query, tk, alpha)
            if hybrid
            else self.similarity_search(db, query, tk)
        )

        # 日付フィルタを適用
        date_range = parse_date_from_query(query)
        date_filtered = False
        date_detected = date_range is not None
        date_no_match = False
        if date_range and matches_all:
            filtered = self._filter_by_date(matches_all, date_range)
            if filtered:
                # 日付でフィルタした結果を優先使用
                matches_all = filtered
                date_filtered = True
            else:
                # 日付は検出されたがマッチするデータがない
                date_no_match = True
            logger.debug(
                "Date filter applied: %s to %s, filtered=%d matches",
                date_range[0],
                date_range[1],
                len(filtered) if filtered else 0,
            )

        t1 = time.time()
        if not matches_all:
            # 空ジェネレータを返す
            no_result_msg = "関連するテキストが見つかりませんでした。"
            if date_range:
                no_result_msg = f"指定された期間（{date_range[0]} 〜 {date_range[1]}）に該当するデータが見つかりませんでした。"
            return {
                "matches": [],
                "meta": {
                    "candidates": 0,
                    "used_context_chunks": 0,
                    "used_context_chars": 0,
                    "date_filter": date_range,
                    "timings_ms": {"retrieval": int((t1 - t0) * 1000.0), "prompt_build": 0},
                },
                "stream_fn": lambda msg=no_result_msg: iter((msg,)),
            }

        # プロンプト用に上限を適用
        use_k = int(context_k or CONTEXT_MAX_CHUNKS)
        selected: List[Dict] = []
        used_chars = 0
        for m in matches_all:
            if len(selected) >= use_k:
                break
            txt = m.get("chunk_text") or ""
            add_len = len(txt) + 128
            if used_chars + add_len > CONTEXT_MAX_CHARS:
                break
            selected.append(m)
            used_chars += add_len

        if not selected:
            head = matches_all[0]
            trimmed = dict(head)
            trimmed["chunk_text"] = (head.get("chunk_text") or "")[: max(200, CONTEXT_MAX_CHARS // 2)]
            selected = [trimmed]

        # 会話形式のプロンプトを生成
        messages = self._build_chat_prompt(query, selected, chat_history)
        t2 = time.time()

        retrieval_s = (t1 - t0)
        prompt_build_s = (t2 - t1)

        def _stream_gen():
            try:
                with self._client.responses.stream(
                    model=COMPLETION_MODEL,
                    input=messages,
                ) as stream:
                    tgen0 = time.time()
                    for event in stream:
                        et = getattr(event, "type", None) or (event.get("type") if isinstance(event, dict) else None)
                        if et == "response.output_text.delta":
                            delta = getattr(event, "delta", None) or (event.get("delta") if isinstance(event, dict) else None)
                            if isinstance(delta, str) and delta:
                                yield delta
                        elif et == "response.error":
                            err = getattr(event, "error", None) or (event.get("error") if isinstance(event, dict) else None)
                            logger.error("Responses stream error: %s", err)
                    tgen1 = time.time()
                    generate_s = max(0.0, tgen1 - tgen0)
                    total_s = retrieval_s + prompt_build_s + generate_s
                    try:
                        logger.debug(
                            "RAG timings (s): retrieval=%.3fs, prompt_build=%.3fs, generate=%.3fs, total=%.3fs (candidates=%d, used_chunks=%d)",
                            retrieval_s,
                            prompt_build_s,
                            generate_s,
                            total_s,
                            len(matches_all),
                            len(selected),
                        )
                    except Exception:
                        pass
            except Exception as exc:
                logger.error("Responses stream failed: %s", exc)
                yield "\n[生成エラー] 回答のストリーミング中にエラーが発生しました。ログを確認してください。"

        meta = {
            "candidates": len(matches_all),
            "used_context_chunks": len(selected),
            "used_context_chars": used_chars,
            "date_filter": {"start": str(date_range[0]), "end": str(date_range[1])} if date_range else None,
            "date_detected": date_detected,
            "date_filtered": date_filtered,
            "date_no_match": date_no_match,
            "limits": {"max_chunks": use_k, "max_chars": CONTEXT_MAX_CHARS},
            # 生成時間はUI側で計測し合算
            "timings_ms": {
                "retrieval": int((t1 - t0) * 1000.0),
                "prompt_build": int((t2 - t1) * 1000.0),
            },
        }

        return {"matches": selected, "meta": meta, "stream_fn": _stream_gen}


rag_service = RAGService()


def get_rag_service() -> RAGService:
    return rag_service
