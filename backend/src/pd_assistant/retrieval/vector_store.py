from pd_assistant.retrieval.interfaces import DocumentChunk, ScoredChunk


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._records: dict[str, tuple[DocumentChunk, list[float]]] = {}

    def upsert(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must be the same")

        for chunk, embedding in zip(chunks, embeddings):    
            self._records[chunk.chunk_id] = (chunk, embedding)



    def search(self, query_embedding: list[float], *, top_k: int, filters: dict[str, str] | None = None, ) -> list[ScoredChunk]:
        scored: list[ScoredChunk] = []

        for chunk, embedding in self._records.values():
            if filters is not None and not self._matches_filters(chunk, filters):
                continue

            score = _dot_product(query_embedding, embedding)
            scored.append(ScoredChunk(chunk=chunk, score=score))

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def _matches_filters(self, chunk: DocumentChunk, filters: dict[str, str]) -> bool:
        for key, expected in filters.items():
            actual = getattr(chunk, key, None)
            if actual != expected:
                return False
        return True


    def _dot_product(a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            raise ValueError("Embedding dimensions must match for scoring")
        return sum(x * y for x, y in zip(a, b))