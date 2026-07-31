from __future__ import annotations

import re
from typing import Iterable, List


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> Iterable[str]:
    """句点ベースのシンプルなチャンク化。"""
    if not text:
        return []

    sentences = [s.strip() for s in re.split(r"(?<=[。．.!?！？])", text) if s and s.strip()]
    if not sentences:
        sentences = [text.strip()]

    # 句点のない長文(擬音注記の連続等)が巨大チャンク化して埋め込みAPIの
    # トークン上限を超えないよう、chunk_sizeを超える文は強制分割する
    bounded: List[str] = []
    for s in sentences:
        if len(s) <= chunk_size:
            bounded.append(s)
        else:
            bounded.extend(s[i : i + chunk_size] for i in range(0, len(s), chunk_size))
    sentences = bounded

    chunks: List[str] = []
    current: List[str] = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_length = len(sentence)
        if current_length + sentence_length <= chunk_size:
            current.append(sentence)
            current_length += sentence_length
            continue

        if current:
            chunks.append("".join(current))

        if chunk_overlap > 0 and chunks:
            overlap_text = chunks[-1][-chunk_overlap:]
            current = [overlap_text, sentence]
            current_length = len(overlap_text) + sentence_length
        else:
            current = [sentence]
            current_length = sentence_length

    if current:
        chunks.append("".join(current))

    return chunks
