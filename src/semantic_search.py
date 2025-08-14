"""
セマンティック検索エンジン

音声文字起こしデータのベクトル化とセマンティック検索機能を提供
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from models import AudioTranscription, get_db

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticSearchEngine:
    """セマンティック検索エンジン"""
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 persist_directory: str = "./chroma_db"):
        """
        初期化
        
        Args:
            model_name: 使用するembeddingモデル（軽量モデル）
            persist_directory: ChromaDBの永続化ディレクトリ
        """
        self.model_name = model_name
        self.persist_directory = persist_directory
        
        # sentence-transformersモデルの初期化
        logger.info(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        
        # ChromaDBクライアントの初期化
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # コレクション名
        self.collection_name = "transcriptions"
        
        # コレクションの取得または作成
        try:
            self.collection = self.chroma_client.get_collection(self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except Exception:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        テキストのembeddingを生成
        
        Args:
            text: 入力テキスト
            
        Returns:
            embedding vector
        """
        return self.embedding_model.encode(text).tolist()
    
    def add_document(self, 
                    document_id: str, 
                    text: str, 
                    metadata: Dict[str, Any]) -> None:
        """
        ドキュメントをベクトルDBに追加
        
        Args:
            document_id: ドキュメントの一意ID
            text: 文字起こしテキスト
            metadata: メタデータ（音声ID、録音時刻、タグなど）
        """
        try:
            embedding = self.generate_embedding(text)
            
            self.collection.add(
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata],
                ids=[document_id]
            )
            
            logger.info(f"Added document {document_id} to vector database")
            
        except Exception as e:
            logger.error(f"Error adding document {document_id}: {str(e)}")
            raise
    
    def search(self, 
              query: str, 
              n_results: int = 10,
              where_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        セマンティック検索を実行
        
        Args:
            query: 検索クエリ
            n_results: 返す結果数
            where_filter: メタデータフィルター
            
        Returns:
            検索結果のリスト
        """
        try:
            query_embedding = self.generate_embedding(query)
            
            # ChromaDBで検索
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
            
            # 結果を整形
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    formatted_results.append({
                        'document': doc,
                        'metadata': metadata,
                        'similarity_score': 1 - distance,  # cosine distanceから類似度に変換
                        'rank': i + 1
                    })
            
            logger.info(f"Found {len(formatted_results)} results for query: {query}")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        コレクションの統計情報を取得
        
        Returns:
            統計情報
        """
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection_name,
                "model_name": self.model_name
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"error": str(e)}

def get_semantic_search_engine() -> SemanticSearchEngine:
    """
    セマンティック検索エンジンのシングルトンインスタンスを取得
    """
    if not hasattr(get_semantic_search_engine, '_instance'):
        get_semantic_search_engine._instance = SemanticSearchEngine()
    return get_semantic_search_engine._instance