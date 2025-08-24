#!/usr/bin/env python3
"""
意味検索機能のテストスクリプト

Usage:
    python scripts/test_semantic_search.py
"""

import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv
from semantic_search import get_semantic_search_engine
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """メイン処理"""
    # 環境変数を読み込み
    load_dotenv()
    
    print("=" * 50)
    print("🔍 意味検索機能のテスト")
    print("=" * 50)
    
    try:
        # セマンティック検索エンジンを初期化
        search_engine = get_semantic_search_engine()
        
        # 統計情報を表示
        stats = search_engine.get_collection_stats()
        print(f"📊 現在のドキュメント数: {stats['total_documents']}")
        print(f"🎯 使用モデル: {stats['model_name']}")
        print()
        
        # テストクエリ
        test_queries = [
            "会議",
            "要約",
            "質問",
            "プロジェクト",
            "スケジュール"
        ]
        
        for query in test_queries:
            print(f"🔍 検索クエリ: '{query}'")
            print("-" * 30)
            
            try:
                # 検索実行
                results = search_engine.search(query, n_results=3)
                
                if results:
                    for i, result in enumerate(results, 1):
                        score = result['similarity_score']
                        audio_id = result['metadata'].get('audio_id', 'Unknown')
                        file_path = result['metadata'].get('file_path', 'Unknown')
                        text_preview = result['document'][:100] + "..." if len(result['document']) > 100 else result['document']
                        
                        print(f"  {i}. 音声ID: {audio_id}")
                        print(f"     ファイル: {file_path}")
                        print(f"     類似度: {score:.3f}")
                        print(f"     内容: {text_preview}")
                        print()
                else:
                    print("  結果なし")
                    print()
                    
            except Exception as e:
                print(f"  ❌ エラー: {str(e)}")
                print()
        
        print("=" * 50)
        print("✅ テスト完了")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()