"""ローカルSQLiteからTurso(libSQL)リモートへデータを移送するスクリプト。

前提:
- `.env` の `DATABASE_URL` が Turso の `sqlite+libsql://` を指していること
- ローカルの元DBファイルは `./audio_transcriptions.db`

実行例:
    uv run python scripts/migrate_sqlite_to_turso.py
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from typing import Any, Dict, Optional

from models import AudioTranscription, Base, get_db  # noqa: F401 - Baseの副作用でテーブル作成


SRC_DB_PATH = "./audio_transcriptions.db"


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def migrate() -> None:
    # 1) ローカルSQLiteから全レコードを取得
    with closing(sqlite3.connect(SRC_DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                "音声ID" AS id,
                "音声ファイルpath" AS file_path,
                "発言人数" AS speaker_count,
                "録音時刻" AS recorded_at,
                "録音時間" AS duration,
                "文字起こしテキスト" AS text,
                "構造化データ" AS structured,
                "タグ" AS tag
            FROM audio_transcriptions
            ORDER BY "音声ID" ASC
            """
        )
        rows = cur.fetchall()

    if not rows:
        print("ローカルSQLiteに移送対象のデータがありません。")
        return

    # 2) リモート(Turso)へUpsert（同一IDが無い場合のみ挿入）
    inserted = 0
    skipped = 0
    db = next(get_db())
    try:
        for r in rows:
            rid = int(r["id"]) if r["id"] is not None else None
            if rid is None:
                skipped += 1
                continue

            exists = db.get(AudioTranscription, rid)
            if exists:
                skipped += 1
                continue

            structured: Optional[Dict[str, Any]] = None
            if r["structured"]:
                try:
                    structured = json.loads(r["structured"]) if isinstance(r["structured"], str) else r["structured"]
                except Exception:
                    structured = None

            row = AudioTranscription(
                音声ID=rid,
                音声ファイルpath=str(r["file_path"] or ""),
                発言人数=int(r["speaker_count"] or 1),
                録音時刻=_parse_dt(r["recorded_at"]) or datetime.utcnow(),
                録音時間=float(r["duration"] or 0.0),
                文字起こしテキスト=str(r["text"] or ""),
                構造化データ=structured,
                タグ=str(r["tag"]) if r["tag"] is not None else None,
            )
            db.add(row)
            inserted += 1

        db.commit()
    finally:
        db.close()

    print(f"移送完了: 追加 {inserted} 件, 既存のためスキップ {skipped} 件")


if __name__ == "__main__":
    migrate()

