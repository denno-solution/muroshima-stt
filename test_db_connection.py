#!/usr/bin/env python3
"""Supabaseデータベース接続テストスクリプト"""

import os
import sys
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# .envファイルを読み込む
load_dotenv()

def test_connection():
    """データベース接続をテスト"""
    db_url = os.getenv('DATABASE_URL')
    
    if not db_url:
        print("❌ エラー: DATABASE_URLが設定されていません")
        print("📝 .envファイルに以下の形式で設定してください：")
        print("DATABASE_URL=postgresql://postgres.[project-ref]:[password]@aws-0-[region].pooler.supabase.com:6543/postgres")
        return False
    
    # SQLite URLの場合は警告
    if db_url.startswith('sqlite'):
        print("⚠️  警告: DATABASE_URLがSQLiteに設定されています")
        print("📝 SupabaseのPostgreSQL URLに変更してください")
        return False
    
    try:
        print(f"🔄 データベースに接続中...")
        print(f"   URL: {db_url.split('@')[1] if '@' in db_url else 'URLが不正です'}")
        
        # エンジンを作成
        engine = create_engine(db_url)
        
        # 接続テスト
        with engine.connect() as conn:
            # PostgreSQLバージョンを確認
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            print(f"✅ 接続成功！")
            print(f"   PostgreSQL: {version.split(',')[0]}")
            
            # テーブルの存在確認
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'audio_transcriptions'
                )
            """))
            table_exists = result.fetchone()[0]
            
            if table_exists:
                print(f"✅ audio_transcriptionsテーブルが存在します")
                
                # レコード数を確認
                result = conn.execute(text("SELECT COUNT(*) FROM audio_transcriptions"))
                count = result.fetchone()[0]
                print(f"   レコード数: {count}")
            else:
                print(f"❌ audio_transcriptionsテーブルが存在しません")
                print(f"📝 create_tables.sqlをSupabaseのSQL Editorで実行してください")
        
        return True
        
    except Exception as e:
        print(f"❌ 接続エラー: {str(e)}")
        print(f"📝 以下を確認してください：")
        print(f"   1. DATABASE_URLが正しく設定されているか")
        print(f"   2. Supabaseプロジェクトが起動しているか")
        print(f"   3. パスワードが正しいか")
        return False

def test_model_integration():
    """SQLAlchemyモデルとの統合テスト"""
    try:
        # modelsモジュールをインポート
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        from models import AudioTranscription, get_db
        
        print("\n🔄 モデル統合テスト中...")
        
        # セッションを取得
        db = next(get_db())
        
        # テーブルの最新レコードを取得
        latest = db.query(AudioTranscription).order_by(
            AudioTranscription.音声ID.desc()
        ).first()
        
        if latest:
            print(f"✅ 最新レコード:")
            print(f"   音声ID: {latest.音声ID}")
            print(f"   ファイル: {latest.音声ファイルpath}")
            print(f"   録音時刻: {latest.録音時刻}")
        else:
            print(f"ℹ️  レコードがまだありません")
        
        db.close()
        return True
        
    except Exception as e:
        print(f"❌ モデル統合エラー: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Supabaseデータベース接続テスト")
    print("=" * 50)
    
    if test_connection():
        test_model_integration()
    
    print("\n✨ テスト完了")