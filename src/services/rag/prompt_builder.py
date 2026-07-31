"""回答生成用プロンプトの組み立て(Responses API形式)。"""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

from services.rag.context_builder import ContextDoc

_SOURCE_LABELS = {"audio": "現場録音", "ceo": "社長音声"}


_CORPUS_DESCRIPTIONS = {
    "audio": "現場の作業録音を文字起こしした「音声DB」",
    "ceo": "社長が録音した音声メモ・打ち合わせを文字起こしした「社長音声DB」",
}


def build_system_prompt(today: Optional[date] = None, corpus: str = "audio") -> str:
    today_str = (today or date.today()).isoformat()
    corpus_desc = _CORPUS_DESCRIPTIONS.get(corpus, _CORPUS_DESCRIPTIONS["audio"])
    return (
        "あなたは射出成形工場の社内アシスタントです。"
        f"{corpus_desc}から検索した内容(コンテキスト)に基づいて質問に答えます。\n"
        f"今日の日付: {today_str}\n"
        "ルール:\n"
        "- 事実は必ずコンテキストに基づき、該当する録音番号 [#n] を出典として示す\n"
        "- コンテキストは音声の自動文字起こしのため、誤変換や言い淀みが含まれうる。文脈から明らかな誤変換は補って解釈してよいが、推測した場合はその旨を付記する\n"
        "- 回答の形式は質問の指定に従う(表形式・報告書形式など)。指定がなければ簡潔な箇条書き\n"
        "- 日付は YYYY-MM-DD 形式で明示する\n"
        "- コンテキストに根拠がない事項は推測せず、「記録には見つからない」と正直に述べる\n"
        "- 会話の文脈を維持し、直前のやり取りとの関連を保つ"
    )


def format_context_block(docs: List[ContextDoc]) -> str:
    blocks: List[str] = []
    for d in docs:
        meta_parts = [f"録音: {d.title}"]
        if d.recorded_date:
            meta_parts.append(f"録音日: {d.recorded_date}")
        if d.tags:
            meta_parts.append(f"タグ: {d.tags}")
        meta_parts.append(_SOURCE_LABELS.get(d.source, d.source))
        if not d.is_full_text:
            meta_parts.append("※関連部分の抜粋")
        header = f"[#{d.n}] " + " / ".join(meta_parts)
        blocks.append(f"{header}\n{d.text}")
    return "\n\n".join(blocks)


def build_chat_messages(
    query: str,
    docs: List[ContextDoc],
    chat_history: Optional[List[Dict]] = None,
    today: Optional[date] = None,
    corpus: str = "audio",
    notes: Optional[List[str]] = None,
) -> List[Dict]:
    """回答生成用メッセージ列を組み立てる。

    notesには検索システム側の補足(期間拡大した・期間内の一部のみ参照している等)を
    渡す。モデルがコンテキストの範囲を誤解して「日付が矛盾する」等と混乱するのを防ぐ。
    """
    messages: List[Dict] = [{"role": "system", "content": build_system_prompt(today, corpus)}]

    if chat_history:
        for msg in chat_history[-10:]:
            role = msg.get("role")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    notes_block = ""
    if notes:
        notes_block = "検索システムからの補足:\n" + "\n".join(f"- {n}" for n in notes) + "\n\n"

    user_prompt = (
        "以下は音声DBから検索した録音の文字起こしです。これに基づいて質問に答えてください。\n\n"
        f"{notes_block}"
        f"{format_context_block(docs)}\n\n"
        f"質問:\n{query}"
    )
    messages.append({"role": "user", "content": user_prompt})
    return messages
