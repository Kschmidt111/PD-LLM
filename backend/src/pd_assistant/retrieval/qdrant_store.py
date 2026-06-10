"""Qdrant-backed vector store."""

import uuid

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from pd_assistant.core.config import get_settings
from pd_assistant.retrieval.interfaces import DocumentChunk, ScoredChunk

_POINT_ID_NAMESPACE = uuid.UUID("f47ac10b-58cc-4372-a567-0e02b2c3d479")


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, chunk_id))


def _chunk_to_payload(chunk: DocumentChunk) -> dict[str, str]:
    return {
        "chunk_id": chunk.chunk_id,
        "source_id": chunk.source_id,
        "text": chunk.text,
        "title": chunk.title,
        "doc_type": chunk.doc_type,
        "version": chunk.version,
        "content_hash": chunk.content_hash,
    }


def _payload_to_chunk(payload: dict[str, str]) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=payload.get("chunk_id", ""),
        source_id=payload.get("source_id", ""),
        text=payload.get("text", ""),
        title=payload.get("title", ""),
        doc_type=payload.get("doc_type", ""),
        version=payload.get("version", ""),
        content_hash=payload.get("content_hash", ""),
    )


class QdrantVectorStore:
    def __init__(self, url: str | None = None, collection: str | None = None) -> None:
        settings = get_settings()
        self._client = QdrantClient(url=url or settings.qdrant_url)
        self._collection = collection or settings.qdrant_collection

    def health_check(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False

    def upsert(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must be the same")
        if not chunks:
            return

        self._ensure_collection(len(embeddings[0]))
        points = [
            PointStruct(
                id=_point_id(chunk.chunk_id),
                vector=embedding,
                payload=_chunk_to_payload(chunk),
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]
        self._client.upsert(collection_name=self._collection, points=points, wait=True)

    def search(self, query_embedding: list[float], *, top_k: int, filters: dict[str, str] | None = None) -> list[ScoredChunk]:
        query_filter = None
        if filters:
            query_filter = Filter(
                must=[
                    FieldCondition(key=key, match=MatchValue(value=value))
                    for key, value in filters.items()
                ]
            )

        response = self._client.query_points(
            collection_name=self._collection,
            query=query_embedding,
            limit=top_k,
            query_filter=query_filter,
        )

        return [
            ScoredChunk(chunk=_payload_to_chunk(point.payload or {}), score=point.score)
            for point in response.points
        ]

    def _ensure_collection(self, vector_size: int) -> None:
        try:
            self._client.get_collection(self._collection)
        except UnexpectedResponse as exc:
            if exc.status_code != 404:
                raise
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
