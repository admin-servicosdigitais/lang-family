"""Passo 2: dimensão fixa sem vocabulário, com hashing de palavras."""

import hashlib
import re

Vector = list[float]

WORDS = [
    "entrega",
    "frete",
    "pix",
    "boleto",
    "prazo",
    "troca",
    "defeito",
    "garantia",
    "reembolso",
    "dados",
    "exclusão",
    "rastreamento",
    "cupom",
    "pedido",
    "nota",
]


def position(word: str, dimension: int) -> int:
    # SHA-256 em vez de hash(): hash() de texto muda a cada execução do Python.
    digest = hashlib.sha256(word.encode()).digest()
    return int.from_bytes(digest[:4], "big") % dimension


def embed(text: str, dimension: int) -> Vector:
    vector = [0.0] * dimension
    for word in re.findall(r"\w+", text.lower()):
        vector[position(word, dimension)] += 1.0
    return vector


def main() -> None:
    print(f"vetor de 'frete grátis' com 8 posições: {embed('frete grátis', 8)}\n")
    print(f"{len(WORDS)} palavras diferentes:")
    for dimension in (8, 64, 1024):
        taken = {position(word, dimension) for word in WORDS}
        print(f"  dimensão {dimension:>4}: {len(WORDS) - len(taken)} colisões")
    print("\nColisão: palavras diferentes na mesma posição parecem iguais para a busca.")
    print(f"'não' e 'nao' coincidem? {embed('não', 64) == embed('nao', 64)}")


if __name__ == "__main__":
    main()
