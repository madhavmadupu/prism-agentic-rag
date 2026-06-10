from __future__ import annotations

from dataclasses import dataclass, field

from langchain.text_splitter import RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    text: str
    metadata: dict
    chunk_id: str | None = None


class DocumentChunker:
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: list[str] | None = None,
    ) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or ["\n\n", "\n", ".", " ", ""],
            length_function=len,
        )

    def chunk_document(
        self,
        text: str,
        base_metadata: dict | None = None,
    ) -> list[Chunk]:
        if not text.strip():
            return []

        metadata = dict(base_metadata or {})
        texts = self._splitter.split_text(text)
        chunks: list[Chunk] = []

        for i, chunk_text in enumerate(texts):
            chunk_meta = {
                **metadata,
                "chunk_index": i,
                "total_chunks": len(texts),
            }
            chunks.append(Chunk(text=chunk_text, metadata=chunk_meta))

        return chunks
