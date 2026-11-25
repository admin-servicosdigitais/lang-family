"""Qdrant: modo local (``:memory:``) ou servidor (URL)."""

import uuid
from collections.abc import Sequence

from qdrant_client import QdrantClient, models

from labs.rag.chunking import Chunk
from labs.rag.embeddings import Vector
from labs.rag.stores.base import Hit, chunk_from, payload_of


def _point_id(chunk_id: str) -> str:
    # O Qdrant só aceita inteiro ou UUID; o UUIDv5 mantém o upsert idempotente.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _match(key: str, value: str) -> models.Filter:
    condition = models.FieldCondition(key=key, match=models.MatchValue(value=value))
    return models.Filter(must=[condition])


class QdrantStore:
    def __init__(self, collection: str, dimension: int, *, location: str = ":memory:") -> None:
        self._client = (
            QdrantClient(location) if location == ":memory:" else QdrantClient(url=location)
        )
        self._name = collection
        if not self._client.collection_exists(collection):
            self._client.create_collection(
                collection,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
            )

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None:
        if not chunks:
            return  # o SDK rejeita um lote vazio
        points = [
            models.PointStruct(id=_point_id(c.id), vector=v, payload=payload_of(c))
            for c, v in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(self._name, points=points, wait=True)

    def query(self, vector: Vector, k: int, *, category: str | None = None) -> list[Hit]:
        response = self._client.query_points(
            self._name,
            query=vector,
            limit=k,
            query_filter=_match("category", category) if category else None,
            with_payload=True,
        )
        return [Hit(chunk_from(point.payload or {}), point.score) for point in response.points]

    def delete_source(self, source: str) -> None:
        selector = models.FilterSelector(filter=_match("source", source))
        self._client.delete(self._name, points_selector=selector, wait=True)

    def count(self) -> int:
        return self._client.count(self._name, exact=True).count

    def close(self) -> None:
        self._client.close()

    def drop(self) -> None:
        self._client.delete_collection(self._name)
