"""Passo 3: conferir as citações e não chamar o modelo quando não há contexto."""

import re
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel, FakeListChatModel

NO_ANSWER = "Não encontrei essa informação na base de conhecimento."
_CITATION = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")


@dataclass(frozen=True)
class Hit:
    source: str
    score: float


@dataclass(frozen=True)
class Answer:
    text: str
    citations: tuple[str, ...] = ()


def cited_numbers(text: str) -> list[int]:
    """Números citados, sem repetição e na ordem em que aparecem."""
    numbers = (int(n) for group in _CITATION.findall(text) for n in re.findall(r"\d+", group))
    return list(dict.fromkeys(numbers))


def answer(model: BaseChatModel, hits: list[Hit], *, min_score: float = 0.3) -> Answer:
    relevant = [hit for hit in hits if hit.score >= min_score]
    if not relevant:
        # Sem contexto não há chamada: nem alucinação, nem custo.
        return Answer(NO_ANSWER)
    text = str(model.invoke("(prompt do passo 2)").text)
    # A citação vem do modelo: só vale se apontar para um trecho que foi enviado.
    citations = tuple(
        relevant[n - 1].source for n in cited_numbers(text) if 1 <= n <= len(relevant)
    )
    return Answer(text, citations)


def main() -> None:
    hits = [Hit("pagamentos.md", 0.82), Hit("entregas.md", 0.41), Hit("garantia.md", 0.12)]
    for reply in ("5% no Pix [1].", "Pix e frete [1, 2].", "Segundo a fonte [7], 50%!"):
        model = FakeListChatModel(responses=[reply])
        print(f"{reply!r:<28} -> citações {answer(model, hits).citations}")

    model = FakeListChatModel(responses=["não deveria ser usado"])
    result = answer(model, [Hit("garantia.md", 0.12)])
    print(f"\ntudo abaixo do corte: {result.text!r} (modelo chamado {model.i} vezes)")


if __name__ == "__main__":
    main()
