from pd_assistant.inference.interfaces import ChatMessage, LLMClient, LLMResponse
from pd_assistant.retrieval.interfaces import (
    DocumentChunk,
    EmbeddingProvider,
    ScoredChunk,
    VectorStore,
)


class StubLLMClient:
    def health_check(self) -> bool:
        return True

    def generate(self, messages: list[ChatMessage]) -> LLMResponse:
        last = messages[-1].content if messages else ""
        return LLMResponse(text=f"stub:{last}")


class StubEmbeddingProvider:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]


class StubVectorStore:
    def __init__(self) -> None:
        self._chunks: list[DocumentChunk] = []

    def upsert(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        del embeddings
        self._chunks.extend(chunks)

    def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
        filters: dict[str, str] | None = None,
    ) -> list[ScoredChunk]:
        del query_embedding, filters
        return [ScoredChunk(chunk=chunk, score=1.0) for chunk in self._chunks[:top_k]]


def test_llm_client_protocol_contract() -> None:
    client: LLMClient = StubLLMClient()
    assert isinstance(client, LLMClient)
    assert client.health_check() is True
    response = client.generate([ChatMessage(role="user", content="hello")])
    assert response.text == "stub:hello"


def test_embedding_provider_protocol_contract() -> None:
    provider: EmbeddingProvider = StubEmbeddingProvider()
    assert isinstance(provider, EmbeddingProvider)
    vectors = provider.embed(["ab", "abcd"])
    assert vectors == [[2.0], [4.0]]


def test_vector_store_protocol_contract() -> None:
    store: VectorStore = StubVectorStore()
    assert isinstance(store, VectorStore)
    chunk = DocumentChunk(
        chunk_id="c1",
        source_id="s1",
        text="alpha-synuclein",
        title="PD biomarker review",
        content_hash="abc123",
    )
    store.upsert([chunk], [[1.0, 0.0]])
    results = store.search([1.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0].chunk.chunk_id == "c1"
    assert results[0].score == 1.0
