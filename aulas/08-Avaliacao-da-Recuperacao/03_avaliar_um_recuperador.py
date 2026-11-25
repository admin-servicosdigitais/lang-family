"""Passo 3: uma função de avaliação que recebe qualquer recuperador."""

import re
from collections.abc import Callable
from dataclasses import dataclass

DOCUMENTS = {
    "pix": "Pagamentos via Pix recebem desconto.",
    "frete": "Pedidos acima de R$ 299 têm frete grátis.",
    "prazo": "A entrega leva até 12 dias úteis.",
    "troca": "Produtos com defeito podem ser trocados.",
}


@dataclass(frozen=True)
class Case:
    question: str
    expected: str


GOLDEN = [
    Case("Pix tem desconto?", "pix"),
    Case("O frete é grátis?", "frete"),
    Case("Quanto tempo demora a entrega?", "prazo"),
    Case("Meu aparelho quebrou, e agora?", "troca"),
]

Retriever = Callable[[str, int], list[str]]


def words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def by_common_words(question: str, k: int) -> list[str]:
    scores = {key: len(words(question) & words(text)) for key, text in DOCUMENTS.items()}
    ranked = sorted(scores, key=lambda key: -scores[key])
    return [key for key in ranked if scores[key] > 0][:k]


def alphabetical(question: str, k: int) -> list[str]:
    """Linha de base ruim de propósito: ignora a pergunta."""
    return sorted(DOCUMENTS)[:k]


def evaluate(retrieve: Retriever, cases: list[Case], *, k: int = 3) -> tuple[float, float]:
    if not cases:
        raise ValueError("informe ao menos um caso de avaliação")
    positions = []
    for case in cases:
        returned = retrieve(case.question, k)
        positions.append(returned.index(case.expected) + 1 if case.expected in returned else None)
    found = [p for p in positions if p is not None]
    return len(found) / len(cases), sum(1 / p for p in found) / len(cases)


def main() -> None:
    for name, retriever in (("linha de base", alphabetical), ("palavras", by_common_words)):
        for k in (1, 3):
            recall, mrr = evaluate(retriever, GOLDEN, k=k)
            print(f"{name:<14} recall@{k}={recall:.2f} MRR={mrr:.2f}")
    misses = [c.question for c in GOLDEN if c.expected not in by_common_words(c.question, 3)]
    print(f"\nperguntas perdidas pela busca por palavras: {misses}")


if __name__ == "__main__":
    main()
