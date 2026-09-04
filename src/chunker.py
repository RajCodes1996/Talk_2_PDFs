"""
chunker.py
Splits extracted PDF text into overlapping chunks suitable for embedding.
Overlap ensures context is not lost at chunk boundaries.
"""

from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[str]:
    """
    Split text into overlapping chunks.

    Args:
        text:          Full extracted document text.
        chunk_size:    Max characters per chunk (~100-150 words).
        chunk_overlap: Characters shared between consecutive chunks
                       so context is not cut off at boundaries.

    Returns:
        List of text chunk strings.
    """
    chunks = _fallback_split_text(text, chunk_size, chunk_overlap)

    # Filter out chunks that are too short to be meaningful
    chunks = [c.strip() for c in chunks if len(c.strip()) > 50]

    return chunks


def _fallback_split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Split text into overlapping chunks without external dependencies.

    This is a lightweight replacement for RecursiveCharacterTextSplitter.
    It prefers paragraph and sentence boundaries, then falls back to spaces.
    """
    if not text.strip():
        return []

    separators = ["\n\n", "\n", ". ", " ", ""]
    chunks: List[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        window_end = min(start + chunk_size, text_length)
        window = text[start:window_end]
        split_at = None

        for separator in separators[:-1]:
            idx = window.rfind(separator)
            if idx != -1:
                candidate = start + idx + len(separator)
                if candidate > start:
                    split_at = candidate
                    break

        if split_at is None:
            split_at = window_end

        chunk = text[start:split_at].strip()
        if chunk:
            chunks.append(chunk)

        if split_at >= text_length:
            break

        start = max(split_at - chunk_overlap, start + 1)

    return chunks


def chunk_with_metadata(text: str, **kwargs) -> List[dict]:
    """
    Returns chunks with index metadata for traceability.
    Useful for citing which part of the document an answer came from.
    """
    chunks = chunk_text(text, **kwargs)
    return [
        {"chunk_id": i, "text": chunk, "char_start": text.find(chunk)}
        for i, chunk in enumerate(chunks)
    ]
