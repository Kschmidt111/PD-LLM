from pd_assistant.retrieval.embeddings import StubEmbeddingProvider
from pd_assistant.retrieval.interfaces import DocumentChunk, ScoredChunk
from pd_assistant.retrieval.service import RetrievalResult, RetrievalService
from pd_assistant.retrieval.vector_store import InMemoryVectorStore


def test_retrieval_service_returns_scored_chunks_and_timing() -> None:
    store = InMemoryVectorStore()
    chunk_a = DocumentChunk(
        chunk_id="c1",
        source_id="s1",
        text="alpha-synuclein",
        doc_type="paper",
    )
    chunk_b = DocumentChunk(
        chunk_id="c2",
        source_id="s2",
        text="PPMI dictionary",
        doc_type="dictionary",
    )
    store.upsert([chunk_a, chunk_b], [[1.0, 0.0], [0.0, 1.0]])

    service = RetrievalService(StubEmbeddingProvider(dimensions=2), store, top_k=2)
    result = service.retrieve("alpha-synuclein")

    assert isinstance(result, RetrievalResult)
    assert len(result.chunks) == 2
    assert all(isinstance(item, ScoredChunk) for item in result.chunks)
    assert all(item.score >= 0 for item in result.chunks)
    assert result.embed_ms >= 0
    assert result.search_ms >= 0
    assert result.total_ms >= 0

    filtered = service.retrieve("anything", filters={"source_id": "s2"})
    assert len(filtered.chunks) == 1
    assert filtered.chunks[0].chunk.chunk_id == "c2"
    assert filtered.chunks[0].chunk.source_id == "s2"
