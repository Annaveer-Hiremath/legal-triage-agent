"""Overlap windows so retrieval can cite a bounded passage."""

from __future__ import annotations

from dataclasses import dataclass

from src.ingestion import ExtractedDocument


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    page: int
    start_char: int
    end_char: int
    document_name: str

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "page": self.page,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "document_name": self.document_name,
        }


def chunk_document(
    doc: ExtractedDocument,
    chunk_size: int = 900,
    overlap: int = 150,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 5)

    chunks: list[Chunk] = []
    counter = 1
    for page_index, page_text in enumerate(doc.pages, start=1):
        text = page_text or ""
        if not text.strip():
            continue
        start = 0
        n = len(text)
        while start < n:
            end = min(start + chunk_size, n)
            piece = text[start:end].strip()
            if piece:
                chunks.append(
                    Chunk(
                        id=f"C{counter}",
                        text=piece,
                        page=page_index,
                        start_char=start,
                        end_char=end,
                        document_name=doc.name,
                    )
                )
                counter += 1
            if end >= n:
                break
            start = end - overlap
    return chunks
