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

def delete_record(record_id: int) -> bool:
    """
    指定されたIDのレコードを削除する
    
    Args:
        record_id: 削除する音声ID
        
    Returns:
        bool: 削除が成功した場合True、失敗した場合False
    """
    db = SessionLocal()
    try:
        record = db.query(AudioTranscription).filter(AudioTranscription.音声ID == record_id).first()
        if record:
            db.delete(record)
            db.commit()
            return True
        return False
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

def delete_all_records() -> int:
    """
    全レコードを削除する
    
    Returns:
        int: 削除されたレコード数
    """
    db = SessionLocal()
    try:
        count = db.query(AudioTranscription).count()
        db.query(AudioTranscription).delete()
        db.commit()
        return count
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()