"""プロンプト組み立て(録音内時刻・VAD基準注記)のテスト。"""

import os
import sys
from datetime import date
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services.rag.context_builder import ContextDoc  # noqa: E402
from services.rag import prompt_builder  # noqa: E402
from services.rag.prompt_builder import (  # noqa: E402
    build_chat_messages,
    build_system_prompt,
    format_context_block,
)


def _doc(n=1, time_basis=None, text="本文です。"):
    return ContextDoc(
        n=n,
        source="audio",
        transcription_id=n,
        title=f"rec{n}.mp3",
        recorded_date="2026-08-01",
        tags=None,
        text=text,
        score=0.9,
        hit_count=1,
        is_full_text=True,
        truncated=False,
        time_basis=time_basis,
    )


class TestSystemPrompt:
    def test_includes_mmss_rule(self):
        prompt = build_system_prompt(today=date(2026, 8, 11))
        assert "MM:SS" in prompt
        assert "録音開始からの経過時間" in prompt

    def test_today_fallback_uses_jst(self, monkeypatch):
        monkeypatch.setattr(prompt_builder, "jst_today", lambda: date(2026, 8, 11))
        assert "今日の日付: 2026-08-11" in build_system_prompt()

    def test_explicit_today(self):
        assert "今日の日付: 2026-07-31" in build_system_prompt(today=date(2026, 7, 31))


class TestVadBasisNote:
    def test_vad_basis_doc_gets_note(self):
        block = format_context_block([_doc(time_basis="vad")])
        assert "無音カット(VAD)後の音声基準" in block
        assert "ズレている可能性" in block

    def test_original_basis_doc_has_no_note(self):
        block = format_context_block([_doc(time_basis="original")])
        assert "無音カット" not in block

    def test_untimed_doc_has_no_note(self):
        block = format_context_block([_doc(time_basis=None)])
        assert "無音カット" not in block

    def test_chat_messages_carry_note(self):
        messages = build_chat_messages(
            "いつ話した？", [_doc(time_basis="vad", text="[00:10〜00:20] 発言。")],
            today=date(2026, 8, 11),
        )
        user_msg = messages[-1]["content"]
        assert "[00:10〜00:20] 発言。" in user_msg
        assert "無音カット(VAD)後の音声基準" in user_msg
