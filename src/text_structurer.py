import os
import json
from typing import Dict, Any, Optional
from google import genai
from google.genai import types
from dotenv import load_dotenv

# .envファイルを読み込む
load_dotenv()

class TextStructurer:
    """Gemini Flash 2.5-liteを使用してテキストをJSON構造化するクラス"""
    
    def __init__(self):
        # Gemini APIキーの設定
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_AI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_AI_API_KEY environment variable is not set")
        
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash-lite"
    
    def structure_text(self, transcribed_text: str) -> Optional[Dict[str, Any]]:
        """文字起こしテキストを構造化データに変換"""
        
        prompt = f"""
以下の文字起こしテキストを解析し、JSON形式で構造化してください。

テキスト:
{transcribed_text}

以下の形式でJSONを生成してください:
{{
  "process_summary": "全体の処理内容の要約",
  "events": [
    {{
      "id": 1,
      "type": "event|parameter_change|action|other",
      "description": "イベントの説明",
      "parameter": "パラメータ名（parameter_changeの場合）",
      "from_value": "変更前の値（parameter_changeの場合）",
      "to_value": "変更後の値（parameter_changeの場合）",
      "name": "イベント名（eventの場合）"
    }}
  ]
}}

注意事項:
- typeは "event"（音などのイベント）、"parameter_change"（パラメータ変更）、"action"（操作）、"other"（その他）から選択
- parameter_changeの場合は、parameter、from_value、to_valueを含める
- eventの場合は、nameを含める
- 数値は可能な限り数値型として扱う
- 日本語で記述してください

JSONのみを返してください。説明文は不要です。
"""
        
        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                    ],
                ),
            ]
            
            generate_content_config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_budget=-1,  # 無制限のthinking
                ),
                response_mime_type="application/json",  # JSONレスポンスを強制
            )
            
            # ストリーミング応答を使って結果を取得
            full_response = ""
            for chunk in self.client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=generate_content_config,
            ):
                if chunk.text:
                    full_response += chunk.text
            
            # レスポンスからJSONを抽出
            json_text = full_response.strip()
            
            # response_mime_type="application/json"により直接JSONが返るが、
            # 念のためコードブロックの除去も行う
            if json_text.startswith("```json"):
                json_text = json_text[7:]
            if json_text.startswith("```"):
                json_text = json_text[3:]
            if json_text.endswith("```"):
                json_text = json_text[:-3]
            
            # JSONパース
            structured_data = json.loads(json_text.strip())
            return structured_data
        except Exception as e:
            print(f"構造化エラー: {e}")
            return None
    
    def extract_tags(self, structured_data: Dict[str, Any]) -> str:
        """構造化データからタグを抽出"""
        tags = []
        
        # process_summaryからキーワードを抽出
        if "process_summary" in structured_data:
            summary = structured_data["process_summary"]
            if "機械" in summary or "設定" in summary:
                tags.append("作業内容")
            if "エラー" in summary or "異常" in summary:
                tags.append("エラー")
            if "メンテナンス" in summary:
                tags.append("メンテナンス")
        
        # eventsから特徴的なタイプを抽出
        if "events" in structured_data:
            event_types = set()
            for event in structured_data["events"]:
                if "type" in event:
                    event_types.add(event["type"])
            
            if "parameter_change" in event_types:
                tags.append("パラメータ変更")
            if "event" in event_types:
                tags.append("イベント発生")
        
        return ", ".join(tags) if tags else "未分類"