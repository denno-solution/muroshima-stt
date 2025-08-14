"""
RAG質問応答システム

セマンティック検索とLLMを組み合わせた質問応答機能を提供
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import google.generativeai as genai
from semantic_search import get_semantic_search_engine

# ログ設定
logger = logging.getLogger(__name__)

class RAGQuestionAnswering:
    """RAG質問応答システム"""
    
    def __init__(self):
        """初期化"""
        # Gemini APIキーの設定
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_AI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY または GOOGLE_AI_API_KEY が設定されていません")
        
        genai.configure(api_key=api_key)
        
        # Geminiモデルの初期化
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # セマンティック検索エンジンを取得
        self.search_engine = get_semantic_search_engine()
        
        logger.info("RAG質問応答システムを初期化しました")
    
    def answer_question(self, 
                       question: str, 
                       max_context_docs: int = 5,
                       min_similarity: float = 0.5) -> Dict[str, Any]:
        """
        質問に対して回答を生成
        
        Args:
            question: 質問文
            max_context_docs: 文脈として使用する最大ドキュメント数
            min_similarity: 使用する最小類似度閾値
            
        Returns:
            回答情報（回答、参照ソース、メタデータ）
        """
        try:
            # 1. セマンティック検索で関連文書を取得
            logger.info(f"質問に対するセマンティック検索実行: {question}")
            search_results = self.search_engine.search(
                query=question, 
                n_results=max_context_docs * 2  # 余分に取得してフィルタリング
            )
            
            if not search_results:
                return {
                    "answer": "申し訳ございませんが、関連する情報が見つかりませんでした。",
                    "sources": [],
                    "confidence": 0.0,
                    "metadata": {"search_count": 0}
                }
            
            # 2. 類似度でフィルタリング
            filtered_results = [
                result for result in search_results 
                if result['similarity_score'] >= min_similarity
            ][:max_context_docs]
            
            if not filtered_results:
                return {
                    "answer": "関連する情報は見つかりましたが、信頼度が低いため回答できません。より具体的な質問をお試しください。",
                    "sources": [],
                    "confidence": 0.0,
                    "metadata": {"search_count": len(search_results)}
                }
            
            # 3. 文脈の構築
            context_parts = []
            sources = []
            
            for i, result in enumerate(filtered_results, 1):
                metadata = result['metadata']
                doc_text = result['document']
                similarity = result['similarity_score']
                
                # 文脈用テキスト
                context_part = f"""
[音声記録 {i}]
日時: {metadata.get('recording_time', '不明')[:19]}
ファイル: {metadata.get('file_path', '不明')}
内容: {doc_text}
"""
                context_parts.append(context_part)
                
                # ソース情報
                sources.append({
                    "audio_id": metadata.get('audio_id', 'Unknown'),
                    "file_path": metadata.get('file_path', 'Unknown'),
                    "recording_time": metadata.get('recording_time', ''),
                    "similarity_score": similarity,
                    "excerpt": doc_text[:200] + "..." if len(doc_text) > 200 else doc_text
                })
            
            context = "\n".join(context_parts)
            
            # 4. LLMでの回答生成
            logger.info("Geminiで回答生成中...")
            answer = self._generate_answer_with_gemini(question, context, sources)
            
            # 5. 信頼度の計算（平均類似度）
            confidence = sum(r['similarity_score'] for r in filtered_results) / len(filtered_results)
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": confidence,
                "metadata": {
                    "search_count": len(search_results),
                    "filtered_count": len(filtered_results),
                    "min_similarity_used": min_similarity
                }
            }
            
        except Exception as e:
            logger.error(f"質問応答エラー: {str(e)}")
            return {
                "answer": f"申し訳ございませんが、回答生成中にエラーが発生しました: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            }
    
    def _generate_answer_with_gemini(self, 
                                   question: str, 
                                   context: str, 
                                   sources: List[Dict[str, Any]]) -> str:
        """
        Geminiを使用して回答を生成
        
        Args:
            question: 質問文
            context: 検索された文脈
            sources: ソース情報
            
        Returns:
            生成された回答
        """
        # プロンプトの構築
        prompt = f"""
あなたは音声議事録の専門アシスタントです。提供された音声記録の内容に基づいて、質問に正確に答えてください。

質問: {question}

参考となる音声記録:
{context}

回答の際は以下のガイドラインに従ってください:
1. 提供された音声記録の内容のみに基づいて回答する
2. 具体的な日時や発言内容があれば明確に示す
3. 複数の記録がある場合は時系列で整理する
4. 推測や憶測は避け、記録にない情報は「記録にありません」と明記する
5. 日本語で自然な文章で回答する
6. 回答の最後に参照した音声記録の数を示す

回答:
"""
        
        try:
            # Geminiで回答生成
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,  # 創造性を抑えて正確性を重視
                    max_output_tokens=1000,
                    top_p=0.8,
                    top_k=40
                )
            )
            
            answer = response.text.strip()
            
            # ソース数の追記
            source_count = len(sources)
            if source_count > 0:
                answer += f"\n\n（参照: {source_count}件の音声記録）"
            
            return answer
            
        except Exception as e:
            logger.error(f"Gemini回答生成エラー: {str(e)}")
            return f"回答生成中にエラーが発生しました: {str(e)}"
    
    def get_conversation_history(self, 
                                topic: str, 
                                max_results: int = 10) -> List[Dict[str, Any]]:
        """
        特定トピックの議論履歴を時系列で取得
        
        Args:
            topic: トピック
            max_results: 最大取得件数
            
        Returns:
            時系列順の議論履歴
        """
        try:
            # セマンティック検索でトピック関連の記録を取得
            results = self.search_engine.search(topic, n_results=max_results)
            
            # 時系列でソート
            sorted_results = sorted(
                results,
                key=lambda x: x['metadata'].get('recording_time', ''),
                reverse=False  # 古い順
            )
            
            return sorted_results
            
        except Exception as e:
            logger.error(f"履歴取得エラー: {str(e)}")
            return []

def get_rag_qa_system() -> RAGQuestionAnswering:
    """
    RAG質問応答システムのシングルトンインスタンスを取得
    """
    if not hasattr(get_rag_qa_system, '_instance'):
        get_rag_qa_system._instance = RAGQuestionAnswering()
    return get_rag_qa_system._instance