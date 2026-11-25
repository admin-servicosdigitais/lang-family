"""Contrato dos bancos vetoriais.

Cada adaptador usa o SDK nativo do banco e entrega o mesmo comportamento:
upsert idempotente, busca por similaridade de cosseno (maior = mais parecido),
filtro por categoria e remoção por documento de origem.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from labs.rag.chunking import Chunk
from labs.rag.embeddings import Vector


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
