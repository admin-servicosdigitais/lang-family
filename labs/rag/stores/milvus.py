"""Milvus: Milvus Lite (arquivo local) ou servidor (URL http)."""

import json
from collections.abc import Sequence

from pymilvus import DataType, MilvusClient

from labs.rag.chunking import Chunk
from labs.rag.embeddings import Vector
from labs.rag.stores.base import Hit, chunk_from, payload_of

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
