"""RAGオーケストレーション。Turso(libSQL)専用。

検索実行はservices/rag/search_service.py、索引の再同期はservices/rag/reconcile.py、
コンテキスト組み立てはservices/rag/context_builder.pyに分離している。
Phase 2(agentic search)ではSearchServiceの各メソッドをLLMのツールとして
公開する想定。
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from openai import OpenAI
from sqlalchemy.orm import Session

from models import (
    AudioTranscriptionChunk,
    CeoTranscriptionChunk,
    EMBEDDING_DIM,
    RAG_FTS_TABLES,
    USE_VECTOR,
)
from sqlalchemy import text as sql_text

from services.rag import chunk_text
from services.rag.context_builder import ContextDoc, build_context_docs
from services.rag.date_utils import DateRange, parse_date_from_query
from services.rag.prompt_builder import build_chat_messages
from services.rag.reconcile import (
    JST,
    cleanup_orphan_chunks,
    fill_recorded_dates,
    find_unindexed,
    normalize_to_jst_date,
    sync_fts,
)
from services.rag.search_service import SearchFilters, SearchService
from services.rag.tokenizer import fts_query_exact, index_tokens, to_fts_text

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "600"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
COMPLETION_MODEL = os.getenv("RAG_COMPLETION_MODEL", "gpt-5.6-luna")
ENABLE_RAG = os.getenv("ENABLE_RAG", "true").lower() in {"1", "true", "yes", "on"}

HYBRID_DEFAULT_ALPHA = float(os.getenv("RAG_HYBRID_ALPHA", "0.6"))
RETRIEVAL_K = int(os.getenv("RAG_RETRIEVAL_K", "100"))

# コンテキストは録音単位で組み立てる
CONTEXT_MAX_DOCS = int(os.getenv("RAG_CONTEXT_MAX_DOCS", "6"))
CONTEXT_MAX_CHARS = int(os.getenv("RAG_CONTEXT_MAX_CHARS", "40000"))
WHOLE_DOC_THRESHOLD = int(os.getenv("RAG_WHOLE_DOC_THRESHOLD", "4000"))

# 「◯◯をまとめて」等で日付だけが手がかりの場合のブラウズ判定に使う、
# 内容語として弱い一般語(この語のバイグラムのみで構成されるクエリはbrowse扱い)
_GENERIC_WORDS = [
    "業務", "記録", "内容", "作業", "状況", "報告", "一覧", "詳細",
    "確認", "データ", "録音", "質問", "回答", "情報", "様子", "結果",
]
_GENERIC_BIGRAMS = {t for w in _GENERIC_WORDS for t in index_tokens(w)}

_HIRAGANA_ONLY = re.compile(r"^[ぁ-んー]+$")


@dataclass
class SearchPlan:
    query: str
    retrieval_text: str
    date_range: Optional[DateRange]
    mode: str  # "search" | "browse"
    sources: Tuple[str, ...]


def _has_content_keywords(text_value: str) -> bool:
    """日付表現を除いたクエリに、検索の手がかりになる内容語があるか。"""
    tokens = index_tokens(text_value)
    for t in tokens:
        if _HIRAGANA_ONLY.match(t):
            continue
        if t in _GENERIC_BIGRAMS:
            continue
        return True
    return False


class RAGService:
    """埋め込み生成・索引管理・検索・回答生成のオーケストレーター。"""

    def __init__(self) -> None:
        self._enabled = bool(USE_VECTOR) and ENABLE_RAG
        self._search = SearchService()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            if self._enabled:
                logger.warning("OPENAI_API_KEY が未設定のため RAG を無効化します")
            self._enabled = False
        self._client = OpenAI() if self._enabled else None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def search(self) -> SearchService:
        return self._search

    # ------------------------------------------------------------------
    # 索引作成
    # ------------------------------------------------------------------
    def index_transcription(self, db: Session, transcription_id: int, text: str) -> bool:
        """現場録音の文字起こしをチャンク化して索引に登録する。"""
        return self._index_generic(db, "audio", transcription_id, text)

    def index_ceo_transcription(self, db: Session, transcription_id: int, text: str) -> bool:
        """社長音声の文字起こしを索引に登録する。"""
        return self._index_generic(db, "ceo", transcription_id, text)

    def _index_generic(self, db: Session, source: str, transcription_id: int, text: str) -> bool:
        if not self.enabled:
            return False

        chunk_model = AudioTranscriptionChunk if source == "audio" else CeoTranscriptionChunk
        fts = RAG_FTS_TABLES[source]

        chunks = list(chunk_text(text, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP))
        if not chunks:
            logger.debug("RAG: チャンクなしのためスキップ (%s id=%s)", source, transcription_id)
            return False

        embeddings = self._embed_texts(chunks)
        if not embeddings or len(embeddings) != len(chunks):
            logger.warning("RAG: 埋め込み生成に失敗したためスキップ (%s id=%s)", source, transcription_id)
            return False

        # recorded_dateの補完(ORMロードは不正JSON列で失敗しうるため生SQL)
        parent_table = "audio_transcriptions" if source == "audio" else "ceo_transcriptions"
        date_source_col = "recorded_at" if source == "ceo" else "created_at"
        row = db.execute(
            sql_text(
                f"SELECT recorded_date, {date_source_col}, created_at FROM {parent_table} WHERE id = :id"
            ),
            {"id": transcription_id},
        ).first()
        if row is not None and not row[0]:
            recorded_date = (
                normalize_to_jst_date(row[1])
                or normalize_to_jst_date(row[2])
                or datetime.now(JST).strftime("%Y-%m-%d")
            )
            db.execute(
                sql_text(f"UPDATE {parent_table} SET recorded_date = :d WHERE id = :id"),
                {"d": recorded_date, "id": transcription_id},
            )

        # 既存チャンク+FTS行を削除してから再作成
        old_ids = [
            r[0]
            for r in db.query(chunk_model.id).filter_by(transcription_id=transcription_id).all()
        ]
        for cid in old_ids:
            db.execute(sql_text(f"DELETE FROM {fts} WHERE rowid = :id"), {"id": cid})
        db.query(chunk_model).filter_by(transcription_id=transcription_id).delete()

        new_chunks = []
        for idx, (piece, embedding) in enumerate(zip(chunks, embeddings)):
            chunk = chunk_model(
                transcription_id=transcription_id,
                chunk_index=idx,
                chunk_text=piece,
                embedding=embedding,
            )
            db.add(chunk)
            new_chunks.append(chunk)
        db.flush()  # idを確定させFTSに登録
        for chunk in new_chunks:
            db.execute(
                sql_text(f"INSERT INTO {fts}(rowid, tokens) VALUES (:id, :tokens)"),
                {"id": chunk.id, "tokens": to_fts_text(chunk.chunk_text)},
            )
        return True

    # ------------------------------------------------------------------
    # 再同期(デスクトップ保存分・社長音声・FTS/日付の差分補完)
    # ------------------------------------------------------------------
    def pending_counts(self, db: Session) -> Dict[str, int]:
        missing = find_unindexed(db)
        return {source: len(ids) for source, ids in missing.items()}

    def reconcile(
        self,
        db: Session,
        embed: bool = True,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict:
        """索引の差分を補完する。embed=Falseなら埋め込み不要の処理のみ。"""
        report: Dict = {}
        report["dates_filled"] = fill_recorded_dates(db)
        report["orphan_chunks_removed"] = cleanup_orphan_chunks(db)
        report["fts"] = sync_fts(db)
        db.commit()

        missing = find_unindexed(db)
        report["unindexed"] = {s: len(ids) for s, ids in missing.items()}
        report["indexed"] = {"audio": 0, "ceo": 0}
        report["errors"] = 0
        if not embed or not self.enabled:
            return report

        total = sum(len(ids) for ids in missing.values())
        done = 0
        for source, ids in missing.items():
            parent_table = "audio_transcriptions" if source == "audio" else "ceo_transcriptions"
            for tid in ids:
                row = db.execute(
                    sql_text(f"SELECT transcript FROM {parent_table} WHERE id = :id"),
                    {"id": tid},
                ).first()
                if row is None or not row[0]:
                    done += 1
                    continue
                try:
                    if self._index_generic(db, source, tid, row[0]):
                        db.commit()
                        report["indexed"][source] += 1
                    else:
                        db.rollback()
                        report["errors"] += 1
                except Exception as exc:
                    db.rollback()
                    report["errors"] += 1
                    logger.error("再同期での索引作成に失敗 (%s id=%s): %s", source, tid, exc)
                done += 1
                if progress_cb:
                    progress_cb(done, total, f"{source} #{tid}")
        return report

    def corpus_stats(self, db: Session) -> Dict[str, Dict]:
        return self._search.corpus_stats(db)

    # ------------------------------------------------------------------
    # 検索計画
    # ------------------------------------------------------------------
    def plan_query(
        self,
        query: str,
        chat_history: Optional[List[Dict]] = None,
        manual_date_range: Optional[Tuple[date, date]] = None,
        sources: Sequence[str] = ("audio",),
        today: Optional[date] = None,
    ) -> SearchPlan:
        if manual_date_range:
            date_range = DateRange(
                manual_date_range[0], manual_date_range[1], "manual", ""
            )
        else:
            date_range = parse_date_from_query(query, today)

        residual = query
        if date_range and date_range.matched_text:
            residual = residual.replace(date_range.matched_text, " ")

        mode = "search"
        if date_range and not _has_content_keywords(residual):
            # 「7/28の作業内容は?」「最近の記録をまとめて」のような、
            # 期間だけが手がかりの質問は期間ブラウズで確実に拾う
            mode = "browse"

        # 短い追問(「具体的に教えて」等)は直前の質問を検索文に含める
        retrieval_text = query
        if chat_history and len(query) <= 12:
            prev_user = next(
                (m.get("content", "") for m in reversed(chat_history) if m.get("role") == "user"),
                "",
            )
            if prev_user:
                retrieval_text = f"{prev_user}\n{query}"

        return SearchPlan(
            query=query,
            retrieval_text=retrieval_text,
            date_range=date_range,
            mode=mode,
            sources=tuple(sources),
        )

    # ------------------------------------------------------------------
    # 回答生成(ストリーミング)
    # ------------------------------------------------------------------
    def answer_stream(
        self,
        db: Session,
        query: str,
        *,
        sources: Sequence[str] = ("audio",),
        manual_date_range: Optional[Tuple[date, date]] = None,
        chat_history: Optional[List[Dict]] = None,
        alpha: Optional[float] = None,
        retrieval_k: Optional[int] = None,
        max_docs: Optional[int] = None,
        today: Optional[date] = None,
    ) -> Dict:
        """検索→コンテキスト組み立てまで実行し、生成はストリーミングで返す。

        戻り値: {"docs": [...], "meta": {...}, "stream_fn": callable}
        """
        if not self.enabled or not self._client:
            return {"docs": [], "meta": {}, "stream_fn": lambda: iter(())}

        alpha = HYBRID_DEFAULT_ALPHA if alpha is None else float(alpha)
        k = int(retrieval_k or RETRIEVAL_K)
        n_docs = int(max_docs or CONTEXT_MAX_DOCS)

        t0 = time.time()
        plan = self.plan_query(
            query,
            chat_history=chat_history,
            manual_date_range=manual_date_range,
            sources=sources,
            today=today,
        )
        filters = SearchFilters(
            date_from=plan.date_range.start if plan.date_range else None,
            date_to=plan.date_range.end if plan.date_range else None,
            sources=plan.sources,
        )

        fallback = None
        widened_term = None
        if plan.mode == "browse":
            hits = self._search.browse_recent(db, filters, max_recordings=n_docs + 2)
            if not hits and plan.date_range and plan.date_range.kind == "recency":
                # 「最近」で30日以内にデータがなければ全期間の直近から
                fallback = "recency_widened"
                hits = self._search.browse_recent(
                    db, SearchFilters(sources=plan.sources), max_recordings=n_docs + 2
                )
        else:
            qvecs = self._embed_texts([plan.retrieval_text])
            qvec = qvecs[0] if qvecs else None
            hits = self._search.hybrid_search(db, plan.retrieval_text, qvec, filters, k, alpha)
            if not hits and plan.date_range and plan.date_range.kind == "recency":
                fallback = "recency_widened"
                filters = SearchFilters(sources=plan.sources)
                hits = self._search.hybrid_search(
                    db, plan.retrieval_text, qvec, filters, k, alpha
                )

            # 「ヒケ」等の引用符付き用語は必須語として扱う。期間内に完全一致が
            # 無く全期間にはある場合、期間を広げて確実に拾う
            if plan.date_range and filters.date_from is not None:
                for term in re.findall(r"[「『\"]([^」』\"]{1,20})[」』\"]", query):
                    match_q = fts_query_exact(term)
                    if not match_q:
                        continue
                    if self._search.keyword_search(db, match_q, filters, 1):
                        continue
                    unfiltered = SearchFilters(sources=plan.sources)
                    if self._search.keyword_search(db, match_q, unfiltered, 1):
                        fallback = "quoted_term_widened"
                        widened_term = term
                        filters = unfiltered
                        hits = self._search.hybrid_search(
                            db, plan.retrieval_text, qvec, filters, k, alpha
                        )
                        break

        t1 = time.time()

        if not hits:
            msg = "関連する録音が見つかりませんでした。"
            if plan.date_range:
                available = self._search.available_date_range(db, plan.sources)
                msg = (
                    f"指定された期間（{plan.date_range.start} 〜 {plan.date_range.end}）に"
                    "該当する録音が見つかりませんでした。"
                )
                if available:
                    msg += f"\n\nデータが存在する期間: {available[0]} 〜 {available[1]}"
            return {
                "docs": [],
                "meta": self._build_meta(plan, fallback, 0, [], 0, t0, t1, t1),
                "stream_fn": lambda m=msg: iter((m,)),
            }

        order = "date" if plan.mode == "browse" else "score"
        docs = build_context_docs(
            db,
            hits,
            max_docs=n_docs,
            max_chars=CONTEXT_MAX_CHARS,
            whole_doc_threshold=WHOLE_DOC_THRESHOLD,
            order=order,
        )
        corpus = "ceo" if tuple(plan.sources) == ("ceo",) else "audio"
        messages = build_chat_messages(query, docs, chat_history, today, corpus=corpus)
        t2 = time.time()

        client = self._client

        def _stream_gen():
            try:
                with client.responses.stream(model=COMPLETION_MODEL, input=messages) as stream:
                    for event in stream:
                        et = getattr(event, "type", None)
                        if et == "response.output_text.delta":
                            delta = getattr(event, "delta", None)
                            if isinstance(delta, str) and delta:
                                yield delta
                        elif et == "response.error":
                            err = getattr(event, "error", None)
                            logger.error("Responses stream error: %s", err)
                            yield f"\n\n⚠️ 回答生成でエラーが発生しました: {err}"
            except Exception as exc:
                logger.error("Responses stream failed: %s", exc, exc_info=True)
                yield (
                    f"\n\n⚠️ 回答の生成に失敗しました（モデル: {COMPLETION_MODEL}）。\n"
                    f"エラー: {exc}"
                )

        used_chars = sum(len(d.text) for d in docs)
        meta = self._build_meta(plan, fallback, len(hits), docs, used_chars, t0, t1, t2)
        meta["widened_term"] = widened_term
        return {"docs": [d.to_dict() for d in docs], "meta": meta, "stream_fn": _stream_gen}

    @staticmethod
    def _build_meta(
        plan: SearchPlan,
        fallback: Optional[str],
        n_candidates: int,
        docs: List[ContextDoc],
        used_chars: int,
        t0: float,
        t1: float,
        t2: float,
    ) -> Dict:
        return {
            "mode": plan.mode,
            "sources": list(plan.sources),
            "date_filter": (
                {
                    "start": plan.date_range.start.isoformat(),
                    "end": plan.date_range.end.isoformat(),
                    "kind": plan.date_range.kind,
                    "matched_text": plan.date_range.matched_text,
                }
                if plan.date_range
                else None
            ),
            "fallback": fallback,
            "candidates": n_candidates,
            "used_docs": len(docs),
            "used_context_chars": used_chars,
            "doc_summaries": [
                {
                    "n": d.n,
                    "title": d.title,
                    "recorded_date": d.recorded_date,
                    "score": d.score,
                    "source": d.source,
                }
                for d in docs
            ],
            "timings_ms": {
                "retrieval": int((t1 - t0) * 1000.0),
                "prompt_build": int((t2 - t1) * 1000.0),
            },
        }

    # ------------------------------------------------------------------
    # 埋め込み
    # ------------------------------------------------------------------
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


rag_service = RAGService()


def get_rag_service() -> RAGService:
    return rag_service
