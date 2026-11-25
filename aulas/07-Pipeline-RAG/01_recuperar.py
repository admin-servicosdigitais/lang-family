"""Passo 1: recuperar os trechos mais próximos de uma pergunta."""

import re

CHUNKS = [
    "Pagamentos via Pix recebem 5% de desconto.",
    "Compras podem ser canceladas em até 7 dias.",
    "Pedidos acima de R$ 299 têm frete grátis.",
]


def similarity(a: str, b: str) -> float:
    """Cosseno entre conjuntos de palavras: um substituto simples dos embeddings (aula 06)."""
    words_a, words_b = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
    if not words_a or not words_b:
        return 0.0
    return float(len(words_a & words_b) / (len(words_a) * len(words_b)) ** 0.5)


def retrieve(question: str, k: int) -> list[tuple[float, str]]:
    scored = [(similarity(question, chunk), chunk) for chunk in CHUNKS]
    return sorted(scored, reverse=True)[:k]


def main() -> None:
    for question in ("Pix tem desconto?", "Qual a previsão do tempo?"):
        print(f"pergunta: {question}")
        for score, chunk in retrieve(question, k=2):
            print(f"  {score:.2f} {chunk}")
    print("\nA busca sempre devolve os k mais próximos, mesmo que nenhum sirva.")


if __name__ == "__main__":
    main()
