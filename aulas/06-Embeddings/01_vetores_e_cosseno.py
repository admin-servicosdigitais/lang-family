"""Passo 1: texto como vetor e a similaridade de cosseno."""

import math
import re

Vector = list[float]

DOCUMENTS = [
    "frete grátis para o sudeste",
    "o frete para o norte é cobrado",
    "garantia cobre defeito de fábrica",
]


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def bag_of_words(text: str, vocabulary: list[str]) -> Vector:
    """Uma posição por palavra conhecida; o valor é quantas vezes ela aparece."""
    tokens = words(text)
    return [float(tokens.count(word)) for word in vocabulary]


def cosine(a: Vector, b: Vector) -> float:
    """Ângulo entre os vetores: 1 = mesma direção, 0 = nada em comum."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norms = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norms if norms else 0.0


def main() -> None:
    vocabulary = sorted({word for text in DOCUMENTS for word in words(text)})
    print(f"vocabulário ({len(vocabulary)} palavras): {vocabulary}\n")
    for question in ("qual o frete para o sudeste", "quanto custa a entrega"):
        query = bag_of_words(question, vocabulary)
        print(f"pergunta: {question}")
        for text in DOCUMENTS:
            print(f"  {cosine(query, bag_of_words(text, vocabulary)):.2f} {text}")
        print()
    print("'entrega' e 'frete' não compartilham letras no vetor: para ele, nada em comum.")


if __name__ == "__main__":
    main()
