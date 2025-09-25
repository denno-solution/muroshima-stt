"""RAG（pgvector）向けの埋め込み生成と保存ロジック。"""

from __future__ import annotations

import logging
import os
import re
from typing import Dict, Iterable, List

from openai import OpenAI
from sqlalchemy import select
from sqlalchemy.orm import Session

from models import AudioTranscription, AudioTranscriptionChunk, EMBEDDING_DIM, USE_VECTOR

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "600"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
COMPLETION_MODEL = os.getenv("RAG_COMPLETION_MODEL", "gpt-4o-mini")
ENABLE_RAG = os.getenv("ENABLE_RAG", "true").lower() in {"1", "true", "yes", "on"}


class RAGService:
    """Supabase/pgvector向けの埋め込み管理。"""

    def __init__(self) -> None:
        self._enabled = bool(USE_VECTOR) and ENABLE_RAG
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

    def answer(self, db: Session, query: str, top_k: int = 5) -> Dict:
        matches = self.similarity_search(db, query, top_k)
        if not matches:
            return {"answer": "関連するテキストが見つかりませんでした。", "matches": []}

        prompt = self._build_prompt(query, matches)
        answer = self._generate_answer(prompt)

        return {"answer": answer, "matches": matches}

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
            logger.error("OpenAI chat_completions API 呼び出しで失敗: %s", exc)
            return "回答生成中にエラーが発生しました。ログを確認してください。"

        choice = response.choices[0] if response.choices else None
        if not choice or not choice.message or not choice.message.content:
            return "回答を生成できませんでした。"
        return choice.message.content.strip()


rag_service = RAGService()


def get_rag_service() -> RAGService:
    return rag_service
