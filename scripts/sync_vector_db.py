#!/usr/bin/env python3
"""
既存の文字起こしデータをベクトルDBに同期するスクリプト

Usage:
    python scripts/sync_vector_db.py
"""

import sys
import os
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv
from semantic_search import get_semantic_search_engine

def main():
    """メイン処理"""
    # 環境変数を読み込み
    load_dotenv()
    
    print("🔄 ベクトルDBとデータベースを同期しています...")
    
    try:
        # セマンティック検索エンジンを初期化
        search_engine = get_semantic_search_engine()
        
        # 統計情報を表示（同期前）
        stats_before = search_engine.get_collection_stats()
        print(f"📊 同期前: {stats_before['total_documents']} ドキュメント")
        
        # データベースと同期
        added_count, updated_count = search_engine.sync_with_database()
        
        # 統計情報を表示（同期後）
        stats_after = search_engine.get_collection_stats()
        print(f"📊 同期後: {stats_after['total_documents']} ドキュメント")
        
        print(f"""
✅ 同期完了
📈 追加: {added_count} ドキュメント
🔄 更新: {updated_count} ドキュメント
🎯 使用モデル: {stats_after['model_name']}
        """)
        
        # 簡単なテスト検索
        print("\n🔍 テスト検索を実行...")
        test_results = search_engine.search("会議", n_results=3)
        
        if test_results:
            print(f"✅ テスト検索成功: {len(test_results)} 件の結果")
            for result in test_results[:2]:  # 上位2件を表示
                score = result['similarity_score']
                audio_id = result['metadata'].get('audio_id', 'Unknown')
                text_preview = result['document'][:100] + "..." if len(result['document']) > 100 else result['document']
                print(f"  • 音声ID {audio_id} (類似度: {score:.3f}): {text_preview}")
        else:
            print("ℹ️ テスト検索結果なし（データがないか、クエリにマッチするものがありません）")
            
    except Exception as e:
        print(f"❌ エラーが発生しました: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()