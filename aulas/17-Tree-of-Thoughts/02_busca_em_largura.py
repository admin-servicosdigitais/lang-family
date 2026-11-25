"""Passo 2: beam search em Python puro, com um avaliador imperfeito."""

from fractions import Fraction
from functools import cache
from itertools import permutations

TARGET = Fraction(24)
Numbers = tuple[Fraction, ...]


def children(numbers: Numbers) -> list[Numbers]:
    result = []
    for a, b in permutations(numbers, 2):
        remaining = list(numbers)
        remaining.remove(a)
        remaining.remove(b)
        values = [a + b, a - b, a * b] + ([a / b] if b else [])
        result += [(*sorted(remaining), value) for value in values]
    return result


@cache
def reachable(numbers: Numbers) -> bool:
    if len(numbers) == 1:
        return numbers[0] == TARGET
    return any(reachable(tuple(sorted(child))) for child in children(numbers))


def rate(numbers: Numbers) -> float:
    """Faz o papel do modelo: acerta os viáveis, mas se encanta com números perto de 24."""
    if reachable(tuple(sorted(numbers))):
        return 6.0
    return 8.0 if any(abs(n - TARGET) <= 2 for n in numbers) else 1.0


def search(start: Numbers, breadth: int) -> bool:
    frontier = [start]
    explored = 0
    while frontier:
        candidates = {tuple(sorted(c)): c for node in frontier for c in children(node)}
        explored += len(candidates)
        if (TARGET,) in candidates:
            print(f"  largura {breadth}: resolveu, explorando {explored} estados")
            return True
        open_nodes = [c for c in candidates.values() if len(c) > 1]
        frontier = sorted(open_nodes, key=rate, reverse=True)[:breadth]  # poda
    print(f"  largura {breadth}: não resolveu, explorando {explored} estados")
    return False


def main() -> None:
    start = tuple(Fraction(n) for n in (2, 7, 8, 9))
    print("números: 2 7 8 9")
    for breadth in (1, 3):
        search(start, breadth)


if __name__ == "__main__":
    main()
