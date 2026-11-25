"""Passo 1: um texto maior que o contexto, com os fatos espalhados."""

FILLER = "Relatório administrativo sem relação com o assunto. " * 20
DOCUMENT = (
    f"O projeto Aurora é liderado por Marina. {FILLER}{FILLER}"
    f"Marina trabalha no escritório de Recife. {FILLER}"
)
CONTEXT_LIMIT = 800  # limite fictício, em caracteres


def split(text: str, size: int) -> list[str]:
    return [text[start : start + size] for start in range(0, len(text), size)]


def main() -> None:
    print(f"documento: {len(DOCUMENT)} caracteres; limite: {CONTEXT_LIMIT}")
    truncated = DOCUMENT[:CONTEXT_LIMIT]
    print("cortando no limite, a cidade aparece?", "Recife" in truncated)

    chunks = split(DOCUMENT, CONTEXT_LIMIT)
    print(f"dividido em {len(chunks)} trechos")
    for number, chunk in enumerate(chunks, start=1):
        facts = [word for word in ("Marina", "Recife") if word in chunk]
        print(f"  trecho {number}: {facts}")
    print("\nNenhum trecho, sozinho, responde: 'onde trabalha quem lidera o Aurora?'")


if __name__ == "__main__":
    main()
