"""Passo 3: normalizar para o produto escalar virar similaridade de cosseno."""

import math
from collections.abc import Sequence

Vector = list[float]


def length(vector: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def normalize(vector: Sequence[float]) -> Vector:
    norm = length(vector)
    return [float(value) / norm for value in vector] if norm else [float(v) for v in vector]


def dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def main() -> None:
    short = [1.0, 1.0, 0.0]
    long = [10.0, 10.0, 0.0]  # mesmo assunto, texto dez vezes maior
    other = [0.0, 1.0, 5.0]

    print("sem normalizar, o tamanho domina o produto escalar:")
    print(f"  curto x longo={dot(short, long):.2f}  longo x outro={dot(long, other):.2f}")
    print(f"  curto x curto={dot(short, short):.2f}")

    a, b, c = normalize(short), normalize(long), normalize(other)
    print("\nnormalizados (comprimento 1), o produto escalar é o cosseno:")
    print(f"  curto x longo={dot(a, b):.2f}  longo x outro={dot(b, c):.2f}")
    print(f"  comprimentos: {length(a):.2f} {length(b):.2f} {length(c):.2f}")

    print(f"\nvetor zero continua zero (evita dividir por zero): {normalize([0.0, 0.0])}")


if __name__ == "__main__":
    main()
