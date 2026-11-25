"""Passo 2: um contrato e um teste que qualquer implementação precisa passar."""

from collections.abc import Callable
from typing import Protocol

Vector = list[float]


class VectorStore(Protocol):
    def upsert(self, ids: list[str], vectors: list[Vector]) -> None: ...

    def query(self, vector: Vector, k: int) -> list[tuple[str, float]]:
        """Devolve (id, similaridade de cosseno): maior = mais parecido."""
        ...


def dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


class SimilarityStore:
    def __init__(self) -> None:
        self.rows: dict[str, Vector] = {}

    def upsert(self, ids: list[str], vectors: list[Vector]) -> None:
        self.rows.update(zip(ids, vectors, strict=True))

    def query(self, vector: Vector, k: int) -> list[tuple[str, float]]:
        scored = [(i, dot(vector, v)) for i, v in self.rows.items()]
        return sorted(scored, key=lambda item: -item[1])[:k]


class DistanceStore(SimilarityStore):
    """Defeito comum: repassar a distância do banco como se fosse pontuação."""

    def query(self, vector: Vector, k: int) -> list[tuple[str, float]]:
        scored = [(i, 1.0 - dot(vector, v)) for i, v in self.rows.items()]
        return sorted(scored, key=lambda item: item[1])[:k]


def check_contract(make: Callable[[], VectorStore]) -> str:
    store = make()
    # Ângulos conhecidos: o cosseno esperado com [1, 0] é 1.0 e 0.8.
    store.upsert(["a", "b"], [[1.0, 0.0], [0.8, 0.6]])
    store.upsert(["a"], [[1.0, 0.0]])
    hits = store.query([1.0, 0.0], 5)
    if [i for i, _ in hits] != ["a", "b"]:
        return f"FALHOU: ordem ou duplicação {hits}"
    scores = [round(score, 3) for _, score in hits]
    if scores != [1.0, 0.8]:
        return f"FALHOU: escala das pontuações {scores}"
    return "passou"


def main() -> None:
    for implementation in (SimilarityStore, DistanceStore):
        print(f"{implementation.__name__:<16} {check_contract(implementation)}")


if __name__ == "__main__":
    main()
