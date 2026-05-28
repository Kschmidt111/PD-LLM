"""A class/function that takes a list of strings and returns a list of float vectors, so later ingestion and retrieval can call it without caring if it’s a stub or Ollama."""




class StubEmbeddingProvider:
    def __init__(self, dimensions: int = 8) -> None:
        self._dimensions = dimensions


    def embed(self, texts: list[str]) -> list[list[float]]:
        """make each vector have length of self._dimensions"""

        embedded_texts: list[list[float]] = []

        for text in texts:
            embedded_text: list[float] = [float(len(text))] * self._dimensions
            embedded_texts.append(embedded_text)

        return embedded_texts