"""Passo 2: MRR, que também considera em que posição o trecho certo veio."""

# Posição (a partir de 1) do trecho correto em cada pergunta; None = não veio.
SYSTEM_A: list[int | None] = [1, 1, 1, None]
SYSTEM_B: list[int | None] = [3, 3, 3, None]


def recall(positions: list[int | None]) -> float:
    return sum(p is not None for p in positions) / len(positions)


def mrr(positions: list[int | None]) -> float:
    """Média de 1/posição. Não encontrado vale zero."""
    return sum(1 / p for p in positions if p is not None) / len(positions)


def main() -> None:
    for name, positions in (("A", SYSTEM_A), ("B", SYSTEM_B)):
        print(
            f"sistema {name}: posições={positions} "
            f"recall@3={recall(positions):.2f} MRR={mrr(positions):.2f}"
        )
    print("\nO recall empata; o MRR mostra que A entrega o certo no topo.")


if __name__ == "__main__":
    main()
