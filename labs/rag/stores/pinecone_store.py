"""Pinecone: um namespace por coleção dentro de um índice com métrica de cosseno."""

from collections.abc import Sequence
from typing import Any

from pinecone import Pinecone, ServerlessSpec

from labs.rag.chunking import Chunk
from labs.rag.embeddings import Vector
from labs.rag.stores.base import Hit, chunk_from, payload_of


def _equals(field: str, value: str) -> dict[str, Any]:
    return {field: {"$eq": value}}


class PineconeStore:
    """Atenção: no serviço real as escritas são eventualmente consistentes."""

    def __init__(self, index: Any, namespace: str) -> None:
        self._index = index
        self._namespace = namespace

    @classmethod
    def from_cloud(
        cls, namespace: str, dimension: int, *, api_key: str, index_name: str = "lang-family"
    ) -> "PineconeStore":
        client = Pinecone(api_key=api_key)
        if not client.has_index(index_name):
            client.create_index(
                index_name,
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                dimension=dimension,
                metric="cosine",
            )
        return cls(client.index(index_name), namespace)

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None:
        records = [
            {"id": c.id, "values": v, "metadata": payload_of(c)}
            for c, v in zip(chunks, vectors, strict=True)
        ]
        self._index.upsert(
            vectors=records, namespace=self._namespace, batch_size=100, show_progress=False
        )

    def query(self, vector: Vector, k: int, *, category: str | None = None) -> list[Hit]:
        response = self._index.query(
            vector=vector,
            top_k=k,
            namespace=self._namespace,
            filter=_equals("category", category) if category else None,
            include_metadata=True,
        )
        return [Hit(chunk_from(dict(m.metadata)), m.score) for m in response.matches]

    def delete_source(self, source: str) -> None:
        self._index.delete(filter=_equals("source", source), namespace=self._namespace)

    def count(self) -> int:
        namespaces = self._index.describe_index_stats().namespaces
        summary = namespaces.get(self._namespace)
        return summary.vector_count if summary else 0

    def close(self) -> None:
        self._index.close()

    def drop(self) -> None:
        if self.count():
            self._index.delete(delete_all=True, namespace=self._namespace)
