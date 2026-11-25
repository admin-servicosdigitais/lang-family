"""Passo 3: um banco de verdade, embarcado, que devolve distância."""

from typing import Any

import chromadb
from chromadb.config import Settings


def main() -> None:
    client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(
        "aula",
        # A métrica é escolhida na criação; o padrão do Chroma não é cosseno.
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,  # os vetores vêm da aplicação
    )
    embeddings: Any = [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]]
    collection.upsert(
        ids=["a", "b", "c"],
        embeddings=embeddings,
        metadatas=[{"category": "x"}, {"category": "x"}, {"category": "y"}],
    )

    query: Any = [[1.0, 0.0]]
    result = collection.query(query_embeddings=query, n_results=3, include=["distances"])
    assert result["distances"] is not None
    for chunk_id, distance in zip(result["ids"][0], result["distances"][0], strict=True):
        print(f"{chunk_id}: distância={distance:.2f} similaridade={1 - distance:.2f}")

    filtered = collection.query(query_embeddings=query, n_results=3, where={"category": "y"})
    print(f"filtro category=y: {filtered['ids'][0]}")

    collection.delete(where={"category": "x"})
    print(f"depois de apagar category=x: {collection.count()} registro")


if __name__ == "__main__":
    main()
