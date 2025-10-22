import logging
import time
import os
from array import array
from datetime import datetime
from typing import Sequence

from dotenv import load_dotenv
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Boolean,
    create_engine,
    text,
)
from sqlalchemy.engine import make_url
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.types import UserDefinedType

try:
    from pgvector.sqlalchemy import Vector  # type: ignore
except Exception:  # pragma: no cover - pgvector未導入環境向け
    Vector = None

# .envファイルを読み込む
load_dotenv()

logger = logging.getLogger(__name__)

Base = declarative_base()

EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))


def _is_libsql(url: str) -> bool:
    try:
        drivername = make_url(url).drivername
        return "libsql" in drivername
    except Exception:
        return False


def _is_postgres(url: str) -> bool:
    try:
        backend = make_url(url).get_backend_name()
        return backend in {"postgresql", "postgres"}
    except Exception:
        return False


def _extract_libsql_auth_token(url: str) -> str | None:
    """DATABASE_URL から authToken を抽出（sqlalchemy-libsql 0.2系向け）。"""
    try:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query or "")
        token = qs.get("authToken") or qs.get("authtoken") or qs.get("auth_token")
        if token and len(token) > 0:
            return token[0]
    except Exception:
        pass
    return None


def _strip_auth_token_from_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query or "")
        # auth に関係するキーを除去
        for k in ["authToken", "authtoken", "auth_token"]:
            if k in qs:
                qs.pop(k, None)
        new_q = urlencode(qs, doseq=True)
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_q, parsed.fragment))
    except Exception:
        return url

class AudioTranscription(Base):
    __tablename__ = 'audio_transcriptions'

    音声ID = Column(Integer, primary_key=True, autoincrement=True)
    音声ファイルpath = Column(String(500), nullable=False)
    発言人数 = Column(Integer, default=1)
    録音時刻 = Column(DateTime, nullable=False, default=datetime.now)
    録音時間 = Column(Float, nullable=False)  # 秒単位
    文字起こしテキスト = Column(Text, nullable=False)
    構造化データ = Column(JSON, nullable=True)
    タグ = Column(String(200), nullable=True)
    chunks = relationship(
        "AudioTranscriptionChunk",
        back_populates="transcription",
        cascade="all, delete-orphan",
    )

    def __repr__(self):
        return f"<AudioTranscription(音声ID={self.音声ID}, ファイル={self.音声ファイルpath})>"


DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///./audio_transcriptions.db')
IS_POSTGRES = _is_postgres(DATABASE_URL)
IS_LIBSQL = _is_libsql(DATABASE_URL)


class LibSQLF32Vector(UserDefinedType):
    """libSQLのF32_BLOBカラムをSQLAlchemyで扱うための型。"""

    cache_ok = True

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def get_col_spec(self, **kw):  # type: ignore[override]
        return f"F32_BLOB({self.dimension})"

    def bind_processor(self, dialect):  # type: ignore[override]
        def process(value):
            if value is None:
                return None
            if isinstance(value, (bytes, bytearray, memoryview)):
                raw = bytes(value)
            else:
                raw = _vector_to_f32_blob(value, self.dimension)
            return raw

        return process

    def result_processor(self, dialect, coltype):  # type: ignore[override]
        def process(value):
            if value is None:
                return None
            return _blob_to_vector(value, self.dimension)

        return process

    @property
    def python_type(self):  # type: ignore[override]
        return list

    def _compiler_dispatch(self, visitor, **kw):  # pragma: no cover - dialect固有
        """型のコンパイル処理をF32_BLOBにフォールバック。"""
        return visitor.visit_user_defined_type(self, **kw)


def _vector_to_f32_blob(values: Sequence[float], dimension: int) -> bytes:
    arr = array('f', (float(v) for v in values))
    length = len(arr)
    if length != dimension:
        if length > dimension:
            arr = arr[:dimension]
        else:
            arr.extend((0.0,) * (dimension - length))
    return arr.tobytes()


def _blob_to_vector(blob: bytes, dimension: int) -> list[float]:
    if isinstance(blob, memoryview):
        data = blob.tobytes()
    else:
        data = bytes(blob)
    arr = array('f')
    arr.frombytes(data)
    if len(arr) > dimension:
        arr = arr[:dimension]
    return list(arr)


if bool(Vector) and IS_POSTGRES:
    VECTOR_BACKEND = "postgres"
elif IS_LIBSQL:
    VECTOR_BACKEND = "libsql"
else:
    VECTOR_BACKEND = None

USE_VECTOR = VECTOR_BACKEND is not None
LIBSQL_VECTOR_INDEX_NAME = "audio_transcription_chunks_embedding_idx"


class AudioTranscriptionChunk(Base):
    __tablename__ = 'audio_transcription_chunks'

    id = Column(Integer, primary_key=True, autoincrement=True)
    transcription_id = Column(
        Integer,
        ForeignKey('audio_transcriptions.音声ID', ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    if VECTOR_BACKEND == "postgres":
        embedding = Column(Vector(EMBEDDING_DIM), nullable=False)
    elif VECTOR_BACKEND == "libsql":
        embedding = Column(LibSQLF32Vector(EMBEDDING_DIM), nullable=False)
    else:
        embedding = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    transcription = relationship("AudioTranscription", back_populates="chunks")


class RAGChatLog(Base):
    __tablename__ = 'rag_chat_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    user_text = Column(Text, nullable=False)
    answer_text = Column(Text, nullable=True)
    contexts = Column(JSON, nullable=True)  # 参照したチャンクやスコアを保持
    used_hybrid = Column(Boolean, default=True, nullable=False)
    alpha = Column(Float, nullable=True)

# データベース接続設定
engine_kwargs = dict(echo=False)
if IS_LIBSQL:
    engine_kwargs["pool_pre_ping"] = True

if IS_LIBSQL:
    connect_args = {}
    token = _extract_libsql_auth_token(DATABASE_URL) or os.getenv("TURSO_AUTH_TOKEN") or os.getenv("LIBSQL_AUTH_TOKEN")
    if token:
        # sqlalchemy-libsql >=0.2.0 は connect_args の "auth_token" を推奨
        connect_args["auth_token"] = token
    url_for_engine = _strip_auth_token_from_url(DATABASE_URL)
    if connect_args:
        engine = create_engine(url_for_engine, connect_args=connect_args, **engine_kwargs)
    else:
        engine = create_engine(url_for_engine, **engine_kwargs)
else:
    engine = create_engine(DATABASE_URL, **engine_kwargs)

if IS_POSTGRES and Vector is not None:
    try:
        with engine.connect() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    except Exception as exc:  # pragma: no cover - 権限不足時の警告
        logger.warning("vector拡張の有効化に失敗: %s", exc)

# テーブル作成（所要時間をログ出力）
_t0 = time.time()
Base.metadata.create_all(bind=engine)
_t1 = time.time()
try:
    logger.debug("DB schema check/create completed in %.3fs", _t1 - _t0)
except Exception:
    pass

# セッション作成
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

if IS_LIBSQL:
    _t2 = time.time()
    try:
        with engine.begin() as connection:
            # ベクトル式インデックス（正しい構文: USING ではなく式）
            connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS "
                    f"{LIBSQL_VECTOR_INDEX_NAME} "
                    "ON audio_transcription_chunks(libsql_vector_idx(embedding))"
                )
            )

            # RAG用の補助インデックス（削除・再作成の高速化）
            connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_chunks_by_transcription "
                    "ON audio_transcription_chunks(transcription_id, chunk_index)"
                )
            )

            # FTS5（ハイブリッド検索用）。コンテンツ同期型 + トリガで追従
            connection.execute(
                text(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS audio_transcription_chunks_fts "
                    "USING fts5(\n"
                    "  chunk_text,\n"
                    "  content='audio_transcription_chunks',\n"
                    "  content_rowid='id',\n"
                    "  tokenize='unicode61'\n"
                    ")"
                )
            )

            # 追従トリガ
            connection.execute(
                text(
                    "CREATE TRIGGER IF NOT EXISTS audio_transcription_chunks_ai "
                    "AFTER INSERT ON audio_transcription_chunks BEGIN\n"
                    "  INSERT INTO audio_transcription_chunks_fts(rowid, chunk_text) VALUES (new.id, new.chunk_text);\n"
                    "END;"
                )
            )

            connection.execute(
                text(
                    "CREATE TRIGGER IF NOT EXISTS audio_transcription_chunks_ad "
                    "AFTER DELETE ON audio_transcription_chunks BEGIN\n"
                    "  INSERT INTO audio_transcription_chunks_fts(audio_transcription_chunks_fts, rowid) VALUES('delete', old.id);\n"
                    "END;"
                )
            )

            connection.execute(
                text(
                    "CREATE TRIGGER IF NOT EXISTS audio_transcription_chunks_au "
                    "AFTER UPDATE ON audio_transcription_chunks BEGIN\n"
                    "  INSERT INTO audio_transcription_chunks_fts(audio_transcription_chunks_fts, rowid) VALUES('delete', old.id);\n"
                    "  INSERT INTO audio_transcription_chunks_fts(rowid, chunk_text) VALUES (new.id, new.chunk_text);\n"
                    "END;"
                )
            )

            # 'rebuild' は高コストのため、必要時のみ実行する
            # 条件: FTSテーブルが空 かつ 基表にデータがある場合のみ
            try:
                fts_count = connection.execute(
                    text("SELECT COUNT(*) FROM audio_transcription_chunks_fts")
                ).scalar()
            except Exception:
                fts_count = 0
            try:
                base_count = connection.execute(
                    text("SELECT COUNT(*) FROM audio_transcription_chunks")
                ).scalar()
            except Exception:
                base_count = 0

            if (fts_count or 0) == 0 and (base_count or 0) > 0:
                connection.execute(
                    text(
                        "INSERT INTO audio_transcription_chunks_fts(audio_transcription_chunks_fts) VALUES('rebuild')"
                    )
                )
    except Exception as exc:  # pragma: no cover - 初期化時の警告
        logger.warning("libSQLの初期化（ベクトル/FTS）に失敗: %s", exc)
    finally:
        _t3 = time.time()
        try:
            logger.debug("libSQL init (indexes/FTS) completed in %.3fs", _t3 - _t2)
        except Exception:
            pass

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
