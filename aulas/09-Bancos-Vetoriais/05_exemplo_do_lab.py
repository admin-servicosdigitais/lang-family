"""Passo 5: três adaptadores do laboratório sob o mesmo contrato.

Reproduz ``labs/rag/stores/`` (base, Chroma, Qdrant e Milvus). Weaviate e Pinecone
ficam no laboratório. O ``main`` é só para a aula.
"""

import json
import tempfile
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings
from pymilvus import DataType, MilvusClient
from qdrant_client import QdrantClient, models

# --- Das aulas 05 e 06 --------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    section: str
    category: str

    @property
    def contextual_text(self) -> str:
        """Texto usado no embedding: o título da seção dá contexto ao trecho."""
        return f"{self.section}\n{self.text}"


Vector = list[float]


# --- Do laboratório: rag/stores/base.py ---------------------------------------------


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class VectorStore(Protocol):
    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None: ...

    def query(self, vector: Vector, k: int, *, category: str | None = None) -> list[Hit]: ...

    def delete_source(self, source: str) -> None: ...

    def count(self) -> int: ...

    def close(self) -> None:
        """Libera conexões; não apaga os dados."""
        ...

    def drop(self) -> None:
        """Apaga a coleção/namespace usada por esta instância."""
        ...


def payload_of(chunk: Chunk) -> dict[str, str]:
    return {
        "chunk_id": chunk.id,
        "text": chunk.text,
        "source": chunk.source,
        "section": chunk.section,
        "category": chunk.category,
    }


def chunk_from(payload: dict[str, object]) -> Chunk:
    return Chunk(
        id=str(payload["chunk_id"]),
        text=str(payload["text"]),
        source=str(payload["source"]),
        section=str(payload["section"]),
        category=str(payload["category"]),
    )


# --- Do laboratório: rag/stores/chroma.py -------------------------------------------


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
        if not chunks:
            return  # o SDK rejeita um lote vazio
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


# --- Do laboratório: rag/stores/qdrant.py -------------------------------------------


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


# --- Do laboratório: rag/stores/milvus.py -------------------------------------------


_FIELDS = ["chunk_id", "text", "source", "section", "category"]


def _equals(field: str, value: str) -> str:
    # json.dumps gera um literal entre aspas e escapado, evitando injeção no filtro.
    return f"{field} == {json.dumps(value, ensure_ascii=False)}"


class MilvusStore:
    def __init__(self, collection: str, dimension: int, *, uri: str) -> None:
        self._client = MilvusClient(uri=uri)
        self._name = collection
        if not self._client.has_collection(collection):
            self._create(dimension)
        self._client.load_collection(collection)

    def _create(self, dimension: int) -> None:
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=512)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dimension)
        schema.add_field("text", DataType.VARCHAR, max_length=8192)
        for field in ("source", "section", "category"):
            schema.add_field(field, DataType.VARCHAR, max_length=512)
        index = MilvusClient.prepare_index_params()
        index.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
        self._client.create_collection(
            self._name, schema=schema, index_params=index, consistency_level="Strong"
        )

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None:
        rows = [payload_of(c) | {"vector": v} for c, v in zip(chunks, vectors, strict=True)]
        self._client.upsert(self._name, data=rows)

    def query(self, vector: Vector, k: int, *, category: str | None = None) -> list[Hit]:
        results = self._client.search(
            self._name,
            data=[vector],
            limit=k,
            filter=_equals("category", category) if category else "",
            output_fields=_FIELDS,
            consistency_level="Strong",
        )
        # Com a métrica COSINE, o campo "distance" já é a similaridade.
        return [Hit(chunk_from(dict(row["entity"])), float(row["distance"])) for row in results[0]]

    def delete_source(self, source: str) -> None:
        self._client.delete(self._name, filter=_equals("source", source))

    def count(self) -> int:
        rows = self._client.query(
            self._name, filter="", output_fields=["count(*)"], consistency_level="Strong"
        )
        return int(rows[0]["count(*)"])

    def close(self) -> None:
        self._client.close()

    def drop(self) -> None:
        self._client.drop_collection(self._name)


# --- Somente da aula ----------------------------------------------------------------


def exercise(store: VectorStore) -> None:
    """O mesmo roteiro para qualquer banco."""
    chunks = [
        Chunk("a.md#0", "frete", "a.md", "Frete", "logistica"),
        Chunk("a.md#1", "prazo", "a.md", "Prazo", "logistica"),
        Chunk("b.md#0", "pix", "b.md", "Pix", "financeiro"),
    ]
    vectors = [[1.0, 0.0, 0.0, 0.0], [0.8, 0.6, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    query = [1.0, 0.0, 0.0, 0.0]
    store.upsert(chunks, vectors)
    store.upsert(chunks, vectors)
    store.upsert([], [])
    hits = store.query(query, 3)
    print(f"  total={store.count()} {[(h.chunk.id, round(h.score, 2)) for h in hits]}")
    print(f"  financeiro: {[h.chunk.id for h in store.query(query, 3, category='financeiro')]}")
    store.delete_source("a.md")
    print(f"  sem a.md  : total={store.count()}")
    store.drop()
    store.close()


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        stores: list[VectorStore] = [
            ChromaStore("aula_lab"),
            QdrantStore("aula_lab", 4),
            MilvusStore("aula_lab", 4, uri=f"{tmp}/m.db"),
        ]
        for store in stores:
            print(type(store).__name__)
            exercise(store)


if __name__ == "__main__":
    main()
