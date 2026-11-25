"""Passo 4: a mesma operação em Chroma, Qdrant e Milvus Lite."""

import json
import tempfile
import uuid
from typing import Any

import chromadb
from chromadb.config import Settings
from pymilvus import MilvusClient
from qdrant_client import QdrantClient, models

VECTORS = {"pedido.md#0": [1.0, 0.0], "pedido.md#1": [0.8, 0.6]}


def chroma() -> list[tuple[str, float]]:
    client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(
        "tres", configuration={"hnsw": {"space": "cosine"}}, embedding_function=None
    )
    embeddings: Any = list(VECTORS.values())
    collection.upsert(ids=list(VECTORS), embeddings=embeddings)
    query: Any = [[1.0, 0.0]]
    result = collection.query(query_embeddings=query, n_results=2, include=["distances"])
    assert result["distances"] is not None
    pairs = zip(result["ids"][0], result["distances"][0], strict=True)
    return [(i, 1.0 - distance) for i, distance in pairs]  # distância -> similaridade


def qdrant() -> list[tuple[str, float]]:
    client = QdrantClient(":memory:")
    params = models.VectorParams(size=2, distance=models.Distance.COSINE)
    client.create_collection("tres", vectors_config=params)
    points = [
        # Só aceita inteiro ou UUID: o UUIDv5 deriva sempre o mesmo id do texto.
        models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, i)), vector=v, payload={"chunk_id": i}
        )
        for i, v in VECTORS.items()
    ]
    client.upsert("tres", points=points, wait=True)
    response = client.query_points("tres", query=[1.0, 0.0], limit=2, with_payload=True)
    return [(str((p.payload or {})["chunk_id"]), p.score) for p in response.points]


def milvus(path: str) -> list[tuple[str, float]]:
    client = MilvusClient(uri=path)
    client.create_collection(
        "tres", dimension=2, id_type="string", max_length=64, metric_type="COSINE"
    )
    client.upsert("tres", data=[{"id": i, "vector": v} for i, v in VECTORS.items()])
    results = client.search("tres", data=[[1.0, 0.0]], limit=2, consistency_level="Strong")
    client.close()
    # Apesar do nome, com COSINE o campo "distance" já é similaridade.
    return [(str(row["id"]), float(row["distance"])) for row in results[0]]


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        for name, hits in (
            ("chroma", chroma()),
            ("qdrant", qdrant()),
            ("milvus", milvus(f"{tmp}/m.db")),
        ):
            print(f"{name:<7} {[(i, round(score, 2)) for i, score in hits]}")

    # Filtro montado com texto: o valor precisa ser escapado.
    hostile = 'x" or category != "x'
    print("\nfiltro inseguro:", f'category == "{hostile}"')
    print("filtro escapado:", f"category == {json.dumps(hostile)}")


if __name__ == "__main__":
    main()
