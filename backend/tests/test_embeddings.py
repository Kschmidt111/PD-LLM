from pd_assistant.retrieval.embeddings import StubEmbeddingProvider


def test_stub_embedding_provider_embeds_texts() -> None:
    provider = StubEmbeddingProvider()
    vectors = provider.embed(["ab", "abcd"])
    assert vectors == [[2.0] * 8, [4.0] * 8]