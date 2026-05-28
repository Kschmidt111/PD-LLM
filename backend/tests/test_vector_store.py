from pd_assistant.retrieval.interfaces import DocumentChunk, ScoredChunk, VectorStore
from pd_assistant.retrieval.vector_store import InMemoryVectorStore


def test_in_memory_vector_store_search_and_filters() -> None:
    store: VectorStore = InMemoryVectorStore()

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

    store.upsert(
        [chunk_a, chunk_b],
        [[1.0, 0.0], [0.0, 1.0]],
    )

    results = store.search([0.9, 0.1], top_k=1)
    assert len(results) == 1
    assert results[0].chunk.chunk_id == "c1"
    assert results[0].score > 0

    filtered = store.search([0.9, 0.1], top_k=5, filters={"source_id": "s2"})
    assert len(filtered) == 1
    assert filtered[0].chunk.chunk_id == "c2"
    assert isinstance(filtered[0], ScoredChunk)
