import logging
import os
from datetime import datetime

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
    create_engine,
    text,
)
from sqlalchemy.engine import make_url
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

try:
    from pgvector.sqlalchemy import Vector  # type: ignore
except Exception:  # pragma: no cover - pgvector未導入環境向け
    Vector = None

# .envファイルを読み込む
load_dotenv()

logger = logging.getLogger(__name__)

Base = declarative_base()

EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))


def _is_postgres(url: str) -> bool:
    try:
        backend = make_url(url).get_backend_name()
        return backend in {"postgresql", "postgres"}
    except Exception:
        return False

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
USE_VECTOR = bool(Vector) and IS_POSTGRES


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
    if USE_VECTOR:
        embedding = Column(Vector(EMBEDDING_DIM), nullable=False)
    else:
        embedding = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    transcription = relationship("AudioTranscription", back_populates="chunks")

# データベース接続設定
engine = create_engine(DATABASE_URL, echo=False)

if IS_POSTGRES and Vector is not None:
    try:
        with engine.connect() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    except Exception as exc:  # pragma: no cover - 権限不足時の警告
        logger.warning("vector拡張の有効化に失敗: %s", exc)

# テーブル作成
Base.metadata.create_all(bind=engine)

# セッション作成
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
