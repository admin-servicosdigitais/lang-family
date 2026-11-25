"""Weaviate (cliente v4, REST + gRPC) com vetores fornecidos pela aplicação."""

from collections.abc import Sequence

import weaviate
from weaviate.classes.config import Configure, DataType, Property, Tokenization, VectorDistances
from weaviate.classes.data import DataObject
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.util import generate_uuid5

from labs.rag.chunking import Chunk
from labs.rag.embeddings import Vector
from labs.rag.stores.base import Hit, chunk_from, payload_of


class WeaviateStore:
    def __init__(
        self,
        collection: str,
        *,
        host: str = "localhost",
        http_port: int = 18080,
        grpc_port: int = 15051,
    ) -> None:
        self._client = weaviate.connect_to_local(host=host, port=http_port, grpc_port=grpc_port)
        # Nomes de coleção do Weaviate começam com maiúscula.
        self._name = collection[0].upper() + collection[1:]
        if not self._client.collections.exists(self._name):
            self._create()
        self._collection = self._client.collections.use(self._name)

    def _create(self) -> None:
        # FIELD = o valor inteiro vira um único token, para o filtro ser exato.
        exact = [
            Property(name=name, data_type=DataType.TEXT, tokenization=Tokenization.FIELD)
            for name in ("chunk_id", "source", "category")
        ]
        free = [Property(name=name, data_type=DataType.TEXT) for name in ("text", "section")]
        index = Configure.VectorIndex.hnsw(distance_metric=VectorDistances.COSINE)
        self._client.collections.create(
            self._name,
            properties=exact + free,
            vector_config=Configure.Vectors.self_provided(vector_index_config=index),
        )

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None:
        if not chunks:
            return  # o SDK rejeita um lote vazio
        objects = [
            DataObject(properties=payload_of(c), uuid=generate_uuid5(c.id), vector=v)
            for c, v in zip(chunks, vectors, strict=True)
        ]
        # A inserção em lote substitui objetos com o mesmo UUID.
        result = self._collection.data.insert_many(objects)
        if result.has_errors:
            raise RuntimeError(f"falha na inserção: {result.errors}")

    def query(self, vector: Vector, k: int, *, category: str | None = None) -> list[Hit]:
        response = self._collection.query.near_vector(
            near_vector=vector,
            limit=k,
            filters=Filter.by_property("category").equal(category) if category else None,
            return_metadata=MetadataQuery(distance=True),
        )
        hits = []
        for item in response.objects:
            assert item.metadata.distance is not None
            chunk = chunk_from(dict(item.properties))
            hits.append(Hit(chunk, 1.0 - item.metadata.distance))
        return hits

    def delete_source(self, source: str) -> None:
        self._collection.data.delete_many(where=Filter.by_property("source").equal(source))

    def count(self) -> int:
        return self._collection.aggregate.over_all(total_count=True).total_count or 0

    def close(self) -> None:
        self._client.close()

    def drop(self) -> None:
        self._client.collections.delete(self._name)
