import time
from dataclasses import dataclass

from pd_assistant.core.config import get_settings
from pd_assistant.retrieval.interfaces import EmbeddingProvider, ScoredChunk, VectorStore


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[ScoredChunk]
    embed_ms: float
    search_ms: float
    total_ms: float


class RetrievalService:
    def __init__(self, embedding_provider: EmbeddingProvider, vector_store: VectorStore, top_k: int | None = None) -> None:
        self._embedder = embedding_provider
        self._vector_store = vector_store
        self._top_k = top_k if top_k is not None else get_settings().top_k

    def retrieve(self, query: str, *, filters: dict[str, str] | None = None) -> RetrievalResult:
        start = time.perf_counter()

        embed_start = time.perf_counter()
        vectors = self._embedder.embed([query])
        query_embedding = vectors[0]
        embed_ms = (time.perf_counter() - embed_start) * 1000.0

        search_start = time.perf_counter()
        chunks = self._vector_store.search(query_embedding, top_k=self._top_k, filters=filters)
        search_end = time.perf_counter()
        search_ms = (search_end - search_start) * 1000.0
        total_ms = (search_end - start) * 1000.0

        return RetrievalResult(
            chunks=chunks,
            embed_ms=embed_ms,
            search_ms=search_ms,
            total_ms=total_ms,
        )
