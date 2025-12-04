from __future__ import annotations

from datetime import date, datetime
from typing import Dict, List, Optional


def build_prompt(query: str, matches: List[Dict]) -> str:
    """非チャット形式の回答用プロンプトを生成。"""
    numbered_context = []
    for i, match in enumerate(matches, start=1):
        meta_parts = []
        if match.get("file_path"):
            meta_parts.append(f"ファイル: {match['file_path']}")
        if match.get("tag"):
            meta_parts.append(f"タグ: {match['tag']}")
        if match.get("recorded_at"):
            meta_parts.append(f"録音時刻: {match['recorded_at']}")
        meta = " / ".join(meta_parts)
        header = f"[#{i} スコア:{match['score']:.3f}] {meta}" if meta else f"[#{i} スコア:{match['score']:.3f}]"
        numbered_context.append(f"{header}\n{match['chunk_text']}")

    context_block = "\n\n".join(numbered_context)

    instructions = (
        "あなたは社内の音声文字起こしデータを根拠に回答する日本語アシスタントです。"
        "事実は必ず下のコンテキスト内から根拠を取り、出典として [#番号] を示してください。"
        "根拠が完全には揃わない場合でも、\"分かっていること\"と\"不足情報\"を分けて簡潔に答えてください。"
        "日付や時刻は可能なら YYYY-MM-DD 形式で明示してください。"
    )

    output_format = (
        "出力は次の3セクションで返してください:\n"
        "1) 回答:\n- 箇条書きで要点のみ（最大5項目）。\n"
        "2) 根拠:\n- 参照した [#番号] と短い引用/要約（1〜3件）。\n"
        "3) 不足情報/前提:\n- 追加で必要な情報や不確実な点。"
    )

    return (
        f"{instructions}\n\n"
        f"コンテキスト（番号付き）:\n{context_block}\n\n"
        f"質問:\n{query}\n\n"
        f"{output_format}"
    )


def build_chat_prompt(
    query: str,
    matches: List[Dict],
    chat_history: Optional[List[Dict]] = None,
) -> List[Dict]:
    """会話履歴込みのプロンプト（Responses API形式）。"""
    numbered_context = []
    for i, match in enumerate(matches, start=1):
        meta_parts = []
        if match.get("file_path"):
            meta_parts.append(f"ファイル: {match['file_path']}")
        if match.get("tag"):
            meta_parts.append(f"タグ: {match['tag']}")
        if match.get("recorded_at"):
            recorded = match["recorded_at"]
            if isinstance(recorded, datetime):
                recorded = recorded.strftime("%Y-%m-%d %H:%M")
            elif isinstance(recorded, date):
                recorded = recorded.strftime("%Y-%m-%d")
            meta_parts.append(f"録音日時: {recorded}")
        meta = " / ".join(meta_parts)
        header = f"[#{i} スコア:{match['score']:.3f}] {meta}" if meta else f"[#{i} スコア:{match['score']:.3f}]"
        numbered_context.append(f"{header}\n{match['chunk_text']}")

    context_block = "\n\n".join(numbered_context)

    system_content = (
        "あなたはRAGベースの社内QAアシスタントです。"
        "事実は必ず与えられたコンテキストに基づき、出典として [#番号] を明記してください。"
        "コンテキスト外の推測はしないでください。足りない点は『不足情報』に列挙します。"
        "文体は簡潔で日本語、箇条書きを優先します。"
        "会話の文脈を維持し、前の質問への回答と関連付けて答えてください。"
    )

    messages = [{"role": "system", "content": system_content}]

    if chat_history:
        recent_history = chat_history[-10:]
        for msg in recent_history:
            role = msg.get("role")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    user_prompt = (
        f"以下のコンテキスト（番号付き）を参照して質問に答えてください。\n\n"
        f"コンテキスト:\n{context_block}\n\n"
        f"質問:\n{query}\n\n"
        f"出力は次の3セクションで返してください:\n"
        f"1) 回答: 箇条書きで要点のみ（最大5項目）。\n"
        f"2) 根拠: 参照した [#番号] と短い引用/要約（1〜3件）。\n"
        f"3) 不足情報/前提: 追加で必要な情報や不確実な点。"
    )
    messages.append({"role": "user", "content": user_prompt})

    return messages
