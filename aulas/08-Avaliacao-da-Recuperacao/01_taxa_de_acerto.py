"""Passo 1: um conjunto de referência e a pergunta "o trecho certo veio entre os k?"."""

# Para cada pergunta: o trecho correto e o que o recuperador devolveu, em ordem.
RESULTS = [
    ("Tem desconto no Pix?", "pix", ["pix", "parcelamento", "reembolso"]),
    ("Posso devolver sem a caixa?", "condicoes", ["defeito", "condicoes", "desistencia"]),
    ("Onde acompanho o pacote?", "rastreamento", ["prazos", "frete", "rastreamento"]),
    ("A garantia cobre água?", "exclusoes", ["cobertura", "defeito", "condicoes"]),
]


def recall_at(k: int) -> float:
    """Fração das perguntas cujo trecho correto aparece nos k primeiros."""
    found = sum(expected in returned[:k] for _, expected, returned in RESULTS)
    return found / len(RESULTS)


def main() -> None:
    for question, expected, returned in RESULTS:
        print(f"{question:<30} esperado={expected:<13} veio={returned}")
    print()
    for k in (1, 2, 3):
        print(f"recall@{k} = {recall_at(k):.2f}")


if __name__ == "__main__":
    main()
