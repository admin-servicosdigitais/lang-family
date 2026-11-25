"""Passo 1: dividir um texto em pedaços de tamanho fixo, e o que dá errado."""

TEXT = (
    "Compras feitas pelo site podem ser canceladas em até 7 dias corridos após o "
    "recebimento, sem necessidade de justificativa. O estorno é integral, incluindo o frete."
)


def split_fixed(text: str, size: int, overlap: int = 0) -> list[str]:
    if not 0 <= overlap < size:
        raise ValueError("a sobreposição precisa ser menor que o tamanho")
    step = size - overlap
    # Para quando o restante já está contido no pedaço anterior.
    return [text[start : start + size] for start in range(0, max(len(text) - overlap, 1), step)]


def main() -> None:
    for overlap in (0, 20):
        chunks = split_fixed(TEXT, size=60, overlap=overlap)
        print(f"tamanho=60 sobreposição={overlap}: {len(chunks)} pedaços")
        for chunk in chunks:
            print(f"  |{chunk}|")
        phrase = "7 dias corridos após o recebimento"
        print(f"  a frase '{phrase}' ficou inteira? {any(phrase in c for c in chunks)}\n")


if __name__ == "__main__":
    main()
