"""Chroma: embarcado (memória ou disco)."""

from collections.abc import Sequence
from typing import Any

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings

from labs.rag.chunking import Chunk
from labs.rag.embeddings import Vector
from labs.rag.stores.base import Hit, chunk_from, payload_of


class ChromaStore:
    def __init__(self, collection: str, *, path: str | None = None) -> None:
        settings = Settings(anonymized_telemetry=False)
        self._client: ClientAPI = (
            chromadb.PersistentClient(path=path, settings=settings)
            if path
            else chromadb.EphemeralClient(settings=settings)
        )
        self._name = collection
        self._collection = self._client.get_or_create_collection(
            collection,
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,
        )

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None:
        metadatas: Any = [payload_of(chunk) for chunk in chunks]
        embeddings: Any = list(vectors)
        self._collection.upsert(
            ids=[chunk.id for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk.text for chunk in chunks],
            metadatas=metadatas,
        )

    def query(self, vector: Vector, k: int, *, category: str | None = None) -> list[Hit]:
        query_embeddings: Any = [vector]
        result = self._collection.query(
            query_embeddings=query_embeddings,
            n_results=k,
            where={"category": category} if category else None,
            include=["metadatas", "distances"],
        )
        assert result["metadatas"] is not None
        assert result["distances"] is not None
        # O Chroma devolve distância de cosseno: similaridade = 1 - distância.
        return [
            Hit(chunk_from(dict(metadata)), 1.0 - distance)
            for metadata, distance in zip(
                result["metadatas"][0], result["distances"][0], strict=True
            )
        ]

    def delete_source(self, source: str) -> None:
        self._collection.delete(where={"source": source})

    def count(self) -> int:
        return self._collection.count()

    def close(self) -> None:
        pass

    def drop(self) -> None:
        self._client.delete_collection(self._name)
