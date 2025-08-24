#!/usr/bin/env python3
"""
既存のRDBレコードをベクトルDBに同期するスクリプト

Usage:
    python scripts/sync_db_to_vector.py
    
    # 全レコードを強制的に再同期（重複チェックなし）
    python scripts/sync_db_to_vector.py --force
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import AudioTranscription
from semantic_search import get_semantic_search_engine
import logging

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def sync_records_to_vector_db(force: bool = False):
    """
    RDBのレコードをベクトルDBに同期
    
    Args:
        force: Trueの場合、重複チェックをスキップして全レコードを再登録
    """
    # 環境変数を読み込み
    load_dotenv()
    
    # データベース接続
    database_url = os.getenv("DATABASE_URL", "sqlite:///./audio_transcriptions.db")
    engine = create_engine(database_url)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # セマンティック検索エンジンを初期化
    logger.info("🔄 セマンティック検索エンジンを初期化しています...")
    search_engine = get_semantic_search_engine()
    
    # 統計情報を表示（同期前）
    stats_before = search_engine.get_collection_stats()
    logger.info(f"📊 同期前: {stats_before['total_documents']} ドキュメント")
    
    # データベースセッションを作成
    session = SessionLocal()
    
    try:
        # 全レコードを取得
        logger.info("📚 データベースからレコードを取得しています...")
        records = session.query(AudioTranscription).all()
        total_records = len(records)
        logger.info(f"📋 {total_records} 件のレコードを処理します")
        
        # 既存のドキュメントIDを取得（重複チェック用）
        existing_ids = set()
        if not force:
            try:
                # ChromaDBのcollectionから既存のIDを取得
                result = search_engine.collection.get()
                if result and 'ids' in result:
                    existing_ids = set(result['ids'])
                    logger.info(f"🔍 {len(existing_ids)} 件の既存ドキュメントを検出")
            except Exception as e:
                logger.warning(f"⚠️ 既存IDの取得に失敗: {str(e)}")
        
        # 同期カウンター
        added_count = 0
        skipped_count = 0
        error_count = 0
        
        # 各レコードを処理
        for i, record in enumerate(records, 1):
            try:
                # ドキュメントID
                doc_id = f"audio_{record.音声ID}"
                
                # 進捗表示
                if i % 10 == 0 or i == total_records:
                    logger.info(f"⏳ 進捗: {i}/{total_records} ({i*100//total_records}%)")
                
                # 重複チェック
                if not force and doc_id in existing_ids:
                    skipped_count += 1
                    continue
                
                # メタデータの準備
                metadata = {
                    "audio_id": record.音声ID,
                    "file_path": record.音声ファイルpath,
                    "recording_time": record.録音時刻.isoformat() if record.録音時刻 else None,
                    "duration": record.録音時間,
                    "speaker_count": record.発言人数,
                    "tags": record.タグ
                }
                
                # 構造化データがある場合は追加
                if record.構造化データ:
                    metadata["has_structured_data"] = True
                    # 構造化データから主要な情報を抽出
                    if isinstance(record.構造化データ, dict):
                        if "要約" in record.構造化データ:
                            metadata["summary"] = record.構造化データ["要約"][:200]  # 最初の200文字
                        if "カテゴリ" in record.構造化データ:
                            metadata["category"] = record.構造化データ["カテゴリ"]
                
                # ベクトルDBに追加
                search_engine.add_document(
                    document_id=doc_id,
                    text=record.文字起こしテキスト,
                    metadata=metadata
                )
                
                added_count += 1
                
            except Exception as e:
                error_count += 1
                logger.error(f"❌ レコード {record.音声ID} の処理中にエラー: {str(e)}")
                continue
        
        # 統計情報を表示（同期後）
        stats_after = search_engine.get_collection_stats()
        
        # 結果サマリー
        logger.info("=" * 50)
        logger.info("✅ 同期完了！")
        logger.info(f"📊 同期後: {stats_after['total_documents']} ドキュメント")
        logger.info(f"➕ 追加: {added_count} 件")
        logger.info(f"⏭️  スキップ（既存）: {skipped_count} 件")
        logger.info(f"❌ エラー: {error_count} 件")
        logger.info(f"🎯 使用モデル: {stats_after['model_name']}")
        logger.info("=" * 50)
        
        # テスト検索
        if added_count > 0 or stats_after['total_documents'] > 0:
            logger.info("\n🔍 テスト検索を実行...")
            test_queries = ["会議", "要約", "質問"]
            
            for query in test_queries:
                try:
                    results = search_engine.search(query, n_results=1)
                    if results:
                        top_result = results[0]
                        score = top_result['similarity_score']
                        audio_id = top_result['metadata'].get('audio_id', 'Unknown')
                        text_preview = top_result['document'][:50] + "..." if len(top_result['document']) > 50 else top_result['document']
                        logger.info(f"  📌 '{query}' → 音声ID {audio_id} (類似度: {score:.3f})")
                    else:
                        logger.info(f"  ℹ️ '{query}' → 結果なし")
                except Exception as e:
                    logger.warning(f"  ⚠️ '{query}' → 検索エラー: {str(e)}")
        
        return added_count, skipped_count, error_count
        
    except Exception as e:
        logger.error(f"❌ 同期中にエラーが発生しました: {str(e)}")
        raise
    finally:
        session.close()

def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description='RDBレコードをベクトルDBに同期')
    parser.add_argument(
        '--force', 
        action='store_true',
        help='既存のドキュメントも含めて全て再登録（重複チェックをスキップ）'
    )
    args = parser.parse_args()
    
    try:
        print("🚀 RDB → ベクトルDB 同期を開始します...")
        print(f"📅 実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if args.force:
            print("⚠️ 強制モード: 全レコードを再登録します")
            response = input("続行しますか? (y/N): ")
            if response.lower() != 'y':
                print("❌ 処理を中止しました")
                sys.exit(0)
        
        sync_records_to_vector_db(force=args.force)
        
    except KeyboardInterrupt:
        print("\n⚠️ ユーザーによって中断されました")
        sys.exit(1)
    except Exception as e:
        print(f"❌ エラーが発生しました: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()