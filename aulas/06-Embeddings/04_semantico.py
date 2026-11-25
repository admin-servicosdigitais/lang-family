"""Passo 4: embeddings semânticos com um modelo local, e o limite de tamanho dele.

Na primeira execução são baixados cerca de 220 MB. Não precisa de chave.
"""

from pathlib import Path

from fastembed import TextEmbedding

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CACHE = Path(__file__).parents[2] / ".cache" / "fastembed"

DOCUMENTS = [
    "Pedidos acima de R$ 299 têm frete grátis.",
    "A garantia cobre defeitos de fabricação por um ano.",
    "Você pode pagar com Pix, boleto ou cartão.",
]


def main() -> None:
    model = TextEmbedding(MODEL, cache_dir=str(CACHE))

    def embed(texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in model.embed(texts)]

    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        return float(dot / (sum(x * x for x in a) ** 0.5 * sum(y * y for y in b) ** 0.5))

    vectors = embed(DOCUMENTS)
    print(f"dimensão: {len(vectors[0])}\n")
    for question in ("A entrega é de graça?", "Meu aparelho veio quebrado", "How can I pay?"):
        query = embed([question])[0]
        ranked = sorted(zip(DOCUMENTS, vectors, strict=True), key=lambda d: -cosine(query, d[1]))
        print(f"pergunta: {question}")
        for text, vector in ranked:
            print(f"  {cosine(query, vector):.2f} {text}")
        print()

    # O tokenizador é um detalhe interno da biblioteca; usado aqui só para contar.
    tokenizer = model.model.tokenizer  # type: ignore[attr-defined]
    print(f"limite do modelo: {tokenizer.truncation['max_length']} tokens")
    filler = "O prazo de entrega depende da cidade e da confirmação do pagamento. " * 30
    a, b = embed([filler + "O reembolso leva dez dias.", filler + "Gatos dormem no telhado."])
    print(f"textos longos com finais diferentes geram o mesmo vetor? {a == b}")
    dense = ("SKU-48213/BR; R$ 1.299,90; NF 000.482.113; " * 12)[:500]
    tokenizer.no_truncation()
    print(f"500 caracteres de códigos e valores = {len(tokenizer.encode(dense).ids)} tokens")


if __name__ == "__main__":
    main()
