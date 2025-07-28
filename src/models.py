from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv

# .envファイルを読み込む
load_dotenv()

Base = declarative_base()

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
    
    def __repr__(self):
        return f"<AudioTranscription(音声ID={self.音声ID}, ファイル={self.音声ファイルpath})>"

# データベース接続設定
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///./audio_transcriptions.db')
engine = create_engine(DATABASE_URL, echo=False)

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