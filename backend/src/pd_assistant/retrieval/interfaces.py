from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    source_id: str
    text: str
    title: str = ""
    doc_type: str = ""
    version: str = ""
    content_hash: str = ""


@dataclass(frozen=True)
class ScoredChunk:
    chunk: DocumentChunk
    score: float


@runtime_checkable
class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""


@runtime_checkable
class VectorStore(Protocol):
    def upsert(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        """Insert or update chunks with precomputed embeddings."""

    def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
        filters: dict[str, str] | None = None,
    ) -> list[ScoredChunk]:
        """Return top_k chunks ranked by similarity to the query embedding."""
