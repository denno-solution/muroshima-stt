"""RAG向けの埋め込み生成と保存ロジック。Turso(libSQL)専用。"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional

from openai import OpenAI
from sqlalchemy.orm import Session

from models import (
    AudioTranscription,
    AudioTranscriptionChunk,
    EMBEDDING_DIM,
    LIBSQL_VECTOR_INDEX_NAME,
    USE_VECTOR,
    VECTOR_BACKEND,
)
from services.rag import (
    LibsqlRetriever,
    chunk_text,
    filter_matches_by_date,
    parse_date_from_query,
    build_chat_prompt,
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


class RAGService:
    """埋め込み管理と検索ロジック。Turso(libSQL)専用で動作。"""

    def __init__(self) -> None:
        self._enabled = bool(USE_VECTOR) and ENABLE_RAG
        self._vector_backend = VECTOR_BACKEND
        self._retriever = LibsqlRetriever(LIBSQL_VECTOR_INDEX_NAME)
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

        chunks = list(chunk_text(text, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP))
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

        if self._vector_backend != "libsql":
            return []

        rows = self._retriever.similarity_search(db, query_vector, top_k)

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
        if not self.enabled:
            return []
        if not ENABLE_FTS:
            return self.similarity_search(db, query, top_k)

        # 埋め込み生成
        qvecs = self._embed_texts([query])
        if not qvecs:
            # ベクトルが使えない場合はFTSのみ
            return self._retriever.fts_only(db, query, top_k)
        qvec = qvecs[0]

        # 候補母集団の件数
        cand_k = max(top_k * HYBRID_CAND_MULT, top_k)

        if self._vector_backend != "libsql":
            return []

        return self._retriever.hybrid_search(db, query, qvec, top_k, cand_k, alpha)

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
            filtered = filter_matches_by_date(matches_all, date_range)
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
        messages = build_chat_prompt(query, selected, chat_history)
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
