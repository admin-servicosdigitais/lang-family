"""Passo 1: o estado do Jogo do 24 e a verificação dos passos em código."""

from fractions import Fraction


def apply(numbers: list[Fraction], a: str, op: str, b: str) -> list[Fraction] | None:
    """Combina dois números disponíveis; devolve None se o passo não for válido."""
    if any(len(t) > 20 or "e" in t.lower() for t in (a, b)):
        return None
    try:
        x, y = Fraction(a), Fraction(b)
        remaining = list(numbers)
        remaining.remove(x)  # o número precisa existir
        remaining.remove(y)
        result = {"+": x + y, "-": x - y, "*": x * y}[op] if op != "/" else x / y
    except (ValueError, KeyError, ZeroDivisionError):
        return None
    return [*sorted(remaining), result]


def main() -> None:
    numbers = [Fraction(n) for n in (3, 3, 8, 8)]
    for step in (("8", "/", "3"), ("9", "+", "1"), ("3", "/", "0"), ("1e999999999", "+", "3")):
        result = apply(numbers, *step)
        print(f"{' '.join(step):<20} -> {None if result is None else [str(n) for n in result]}")

    # A solução de 3 3 8 8 exige frações exatas: 8 / (3 - 8/3) = 24.
    state: list[Fraction] | None = numbers
    for step in (("8", "/", "3"), ("3", "-", "8/3"), ("8", "/", "1/3")):
        assert state is not None
        state = apply(state, *step)
        print(f"  {' '.join(step):<14} {[str(n) for n in state or []]}")
    print(f"com ponto flutuante: 8 / (3 - 8/3) = {8 / (3 - 8 / 3)!r}")


if __name__ == "__main__":
    main()
