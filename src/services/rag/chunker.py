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
