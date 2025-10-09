"""ローカルSQLite(`./audio_transcriptions.db`) から
Turso(libSQL) リモート(DB URLは`.env`の`DATABASE_URL`)へデータ移送し、
RAG用のチャンク+埋め込みをバックフィルするワンショットスクリプト。

注意: `.env` はシェルから `source .env` して実行してください。
例) set -a; source .env; set +a; PYTHONPATH=src uv run python scripts/migrate_to_turso_libsql.py
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from array import array
from contextlib import closing
from datetime import datetime
from typing import Iterable, List, Tuple

from openai import OpenAI

# libsql Pythonクライアント (uv add libsql 済み)
from libsql.libsql import connect as libsql_connect


SRC_DB_PATH = "./audio_transcriptions.db"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))

VECTOR_INDEX_NAME = "audio_transcription_chunks_embedding_idx"


def _parse_turso_from_database_url() -> Tuple[str, str]:
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        raise SystemExit("DATABASE_URL が設定されていません。")

    # 例: sqlite+libsql://<host>/?secure=true&authToken=xxxxx
    m = re.match(r"^sqlite\+libsql://([^/?#]+)(/)?\?(.*)$", db_url)
    if not m:
        raise SystemExit("DATABASE_URL が Turso(libSQL) の形式ではありません。")
    host = m.group(1)
    qs = m.group(3)
    token = ""
    for part in qs.split("&"):
        if part.startswith("authToken="):
            token = part.split("=", 1)[1]
            break
    if not token:
        raise SystemExit("DATABASE_URL に authToken が含まれていません。")

    remote = f"libsql://{host}"
    return remote, token


def _chunk_text(text: str) -> Iterable[str]:
    if not text:
        return []
    sentences = [s.strip() for s in re.split(r"(?<=[。．.!?！？])", text) if s and s.strip()]
    if not sentences:
        sentences = [text.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        L = len(s)
        if current_len + L <= CHUNK_SIZE:
            current.append(s)
            current_len += L
            continue
        if current:
            chunks.append("".join(current))
        if CHUNK_OVERLAP > 0 and chunks:
            ov = chunks[-1][-CHUNK_OVERLAP:]
            current = [ov, s]
            current_len = len(ov) + L
        else:
            current = [s]
            current_len = L
    if current:
        chunks.append("".join(current))
    return chunks


def _to_f32_blob(vec: List[float], dim: int) -> bytes:
    arr = array("f", (float(v) for v in vec))
    if len(arr) != dim:
        if len(arr) > dim:
            arr = arr[:dim]
        else:
            arr.extend([0.0] * (dim - len(arr)))
    return arr.tobytes()


def ensure_remote_schema(conn) -> None:
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")
    # テーブル作成はまとめて実行
    cur.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS audio_transcriptions (
            "音声ID" INTEGER PRIMARY KEY AUTOINCREMENT,
            "音声ファイルpath" VARCHAR(500) NOT NULL,
            "発言人数" INTEGER DEFAULT 1,
            "録音時刻" DATETIME NOT NULL,
            "録音時間" FLOAT NOT NULL,
            "文字起こしテキスト" TEXT NOT NULL,
            "構造化データ" JSON,
            "タグ" VARCHAR(200)
        );

        CREATE TABLE IF NOT EXISTS audio_transcription_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transcription_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding F32_BLOB({EMBEDDING_DIM}) NOT NULL,
            created_at DATETIME NOT NULL,
            FOREIGN KEY(transcription_id) REFERENCES audio_transcriptions ("音声ID") ON DELETE CASCADE
        );
        """
    )
    # ベクトル/FTSインデックス等（未対応のサーバーでは失敗するため握りつぶす）
    try:
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS {VECTOR_INDEX_NAME} ON audio_transcription_chunks(libsql_vector_idx(embedding))"
        )
    except Exception:
        pass

    # 補助インデックス
    try:
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_by_transcription ON audio_transcription_chunks(transcription_id, chunk_index)"
        )
    except Exception:
        pass

    # FTS5（ハイブリッド検索用）
    try:
        cur.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS audio_transcription_chunks_fts
            USING fts5(
              chunk_text,
              content='audio_transcription_chunks',
              content_rowid='id',
              tokenize='unicode61'
            )
            """
        )
        # 追従トリガ
        cur.executescript(
            """
            CREATE TRIGGER IF NOT EXISTS audio_transcription_chunks_ai
            AFTER INSERT ON audio_transcription_chunks BEGIN
              INSERT INTO audio_transcription_chunks_fts(rowid, chunk_text) VALUES (new.id, new.chunk_text);
            END;

            CREATE TRIGGER IF NOT EXISTS audio_transcription_chunks_ad
            AFTER DELETE ON audio_transcription_chunks BEGIN
              INSERT INTO audio_transcription_chunks_fts(audio_transcription_chunks_fts, rowid) VALUES('delete', old.id);
            END;

            CREATE TRIGGER IF NOT EXISTS audio_transcription_chunks_au
            AFTER UPDATE ON audio_transcription_chunks BEGIN
              INSERT INTO audio_transcription_chunks_fts(audio_transcription_chunks_fts, rowid) VALUES('delete', old.id);
              INSERT INTO audio_transcription_chunks_fts(rowid, chunk_text) VALUES (new.id, new.chunk_text);
            END;
            """
        )
        # 初期同期
        cur.execute("INSERT INTO audio_transcription_chunks_fts(audio_transcription_chunks_fts) VALUES('rebuild')")
    except Exception:
        pass
    conn.commit()


def copy_transcriptions(src_path: str, dst_conn) -> int:
    with closing(sqlite3.connect(src_path)) as src:
        src.row_factory = sqlite3.Row
        cur = src.cursor()
        cur.execute(
            """
            SELECT "音声ID" AS id, "音声ファイルpath" AS file_path, "発言人数" AS speaker_count,
                   "録音時刻" AS recorded_at, "録音時間" AS duration,
                   "文字起こしテキスト" AS text, "構造化データ" AS structured, "タグ" AS tag
            FROM audio_transcriptions
            ORDER BY "音声ID" ASC
            """
        )
        rows = cur.fetchall()
    if not rows:
        return 0

    dcur = dst_conn.cursor()
    inserted = 0
    for r in rows:
        rid = int(r["id"]) if r["id"] is not None else None
        if rid is None:
            continue
        # 既存確認
        dcur.execute("SELECT 1 FROM audio_transcriptions WHERE \"音声ID\"=?", (rid,))
        if dcur.fetchone():
            continue
        # JSONは文字列ならそのまま、libSQL側で解釈される
        dcur.execute(
            """
            INSERT INTO audio_transcriptions (
                "音声ID", "音声ファイルpath", "発言人数", "録音時刻", "録音時間",
                "文字起こしテキスト", "構造化データ", "タグ"
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rid,
                r["file_path"] or "",
                int(r["speaker_count"] or 1),
                (r["recorded_at"] or datetime.utcnow().isoformat(sep=" ")),
                float(r["duration"] or 0.0),
                r["text"] or "",
                r["structured"],
                r["tag"],
            ),
        )
        inserted += 1
    dst_conn.commit()
    return inserted


def backfill_chunks(dst_conn, openai_client: OpenAI) -> Tuple[int, int]:
    cur = dst_conn.cursor()
    cur.execute(
        "SELECT \"音声ID\", COALESCE(\"文字起こしテキスト\", '') AS t FROM audio_transcriptions ORDER BY \"音声ID\""
    )
    rows = cur.fetchall()
    total_chunks = 0
    processed_trans = 0
    for rid, text in rows:
        text = text or ""
        chunks = list(_chunk_text(text))
        # 既存削除
        cur.execute("DELETE FROM audio_transcription_chunks WHERE transcription_id=?", (rid,))
        if not chunks:
            dst_conn.commit()
            processed_trans += 1
            continue
        # 埋め込み
        resp = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=chunks)
        if not getattr(resp, "data", None):
            continue
        for idx, item in enumerate(resp.data):
            emb = getattr(item, "embedding", None)
            if not emb:
                continue
            blob = _to_f32_blob(list(emb), EMBEDDING_DIM)
            cur.execute(
                """
                INSERT INTO audio_transcription_chunks
                    (transcription_id, chunk_index, chunk_text, embedding, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (rid, idx, chunks[idx], blob, datetime.utcnow().isoformat(sep=" ")),
            )
            total_chunks += 1
        dst_conn.commit()
        processed_trans += 1
    return processed_trans, total_chunks


def main() -> None:
    remote_url, token = _parse_turso_from_database_url()
    # OpenAIキー確認
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY が未設定です。")

    # リモート接続
    with libsql_connect(remote_url, auth_token=token) as rconn:
        ensure_remote_schema(rconn)
        inserted = copy_transcriptions(SRC_DB_PATH, rconn)
        print(f"転送: audio_transcriptions 追加 {inserted} 件")

        client = OpenAI()
        processed, total_chunks = backfill_chunks(rconn, client)
        print(f"バックフィル: transcription {processed} 件, 生成チャンク {total_chunks} 件")

        # 検証出力
        cur = rconn.cursor()
        cur.execute("SELECT COUNT(*) FROM audio_transcriptions")
        n_tr = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM audio_transcription_chunks")
        n_ck = cur.fetchone()[0]
        print(f"リモート件数: transcriptions={n_tr}, chunks={n_ck}")


if __name__ == "__main__":
    main()
