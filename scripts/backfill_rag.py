"""既存の音声文字起こしをRAG索引用にバックフィルするスクリプト。"""

from __future__ import annotations

from typing import Iterable

from models import AudioTranscription, get_db
from services.rag_service import get_rag_service


BATCH_SIZE = 50


def _batched(iterable: Iterable[AudioTranscription], size: int):
    batch: list[AudioTranscription] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def main() -> None:
    rag = get_rag_service()
    if not rag.enabled:
        raise SystemExit("RAGが有効化されていません。DATABASE_URLやOPENAI_API_KEYを確認してください。")

    db = next(get_db())
    try:
        records = db.query(AudioTranscription).order_by(AudioTranscription.id).all()
        if not records:
            print("バックフィル対象のレコードはありません。")
            return

        total = len(records)
        processed = 0
        for chunk in _batched(records, BATCH_SIZE):
            for row in chunk:
                text = row.transcript or ""
                rag.index_transcription(db, row.id, text)
                processed += 1
            db.commit()
            print(f"{processed}/{total} 件を処理しました")
        print("バックフィルが完了しました。")
    finally:
        db.close()


if __name__ == "__main__":
    main()
