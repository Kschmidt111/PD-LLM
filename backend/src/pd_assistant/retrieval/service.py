from dataclasses import dataclass
from pd_assistant.retrieval.interfaces import DocumentChunk, ScoredChunk, EmbeddingProvider, VectorStore
import time

@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[ScoredChunk]
    embed_ms: float
    search_ms: float
    total_ms: float

class RetrievalService:
    def __init__(self, embedding_provider: EmbeddingProvider, vector_store: VectorStore, top_k: int = 5) -> None:
        self._ebedder = embedding_provider
        self._vector_store = vector_store
        if top_k is None:
            self._top_k = get_settings().top_k
        else:
            self._top_k = top_k

    def retrieve(self, query: str, filters: dict[str, str] | None = None) -> RetrievalResult:
        start = time.perf_counter()
        embed_start = time.perf_counter()
        vectors = self._ebedder.embed([query])
        query_embedding = vectors[0]
        embed_ms = (time.perf_counter() - embed_start) * 1000

        search_start = time.perf_counter()
        chunks = self._vector_store.search(query_embedding, top_k=self._top_k, filters=filters)
        search_ms = (time.perf_counter() - search_start) * 1000

        total_ms = (time.perf_counter() - start) * 1000
        return RetrievalResult(chunks=chunks, embed_ms=embed_ms, search_ms=search_ms, total_ms=total_ms)

        