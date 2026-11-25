"""Passo 1: o que um banco vetorial faz, em uma versão em memória."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Record:
    id: str
    category: str
    vector: tuple[float, float]  # já normalizado (aula 06)


class MemoryStore:
    def __init__(self) -> None:
        self.records: dict[str, Record] = {}

    def upsert(self, records: list[Record]) -> None:
        # Mesmo id substitui: reingerir não duplica.
        self.records.update({record.id: record for record in records})

    def query(
        self, vector: tuple[float, float], k: int, *, category: str | None = None
    ) -> list[tuple[str, float]]:
        scored = [
            (r.id, vector[0] * r.vector[0] + vector[1] * r.vector[1])
            for r in self.records.values()
            if category is None or r.category == category  # filtro antes da ordenação
        ]
        return sorted(scored, key=lambda item: -item[1])[:k]


def main() -> None:
    store = MemoryStore()
    records = [
        Record("a", "logistica", (1.0, 0.0)),
        Record("b", "logistica", (0.8, 0.6)),
        Record("c", "financeiro", (0.0, 1.0)),
    ]
    store.upsert(records)
    store.upsert(records)
    print(f"registros: {len(store.records)}")
    print(f"top 2                : {store.query((1.0, 0.0), 2)}")
    print(f"top 2 do financeiro  : {store.query((1.0, 0.0), 2, category='financeiro')}")
    print("\nComparar com todos é exato, mas o custo cresce junto com a base.")


if __name__ == "__main__":
    main()
