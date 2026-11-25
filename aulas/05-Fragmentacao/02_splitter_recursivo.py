"""Passo 2: o splitter recursivo respeita parágrafos, linhas e palavras."""

from langchain_text_splitters import RecursiveCharacterTextSplitter

TEXT = (
    "Compras feitas pelo site podem ser canceladas em até 7 dias.\n"
    "O estorno é integral, incluindo o frete.\n\n"
    "Produtos com defeito podem ser trocados em até 90 dias após a entrega. "
    "É preciso enviar fotos do problema pelo chat antes de despachar o item."
)


def show(size: int, overlap: int) -> None:
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    chunks = splitter.split_text(TEXT)
    print(f"tamanho={size} sobreposição={overlap}: {len(chunks)} pedaços")
    for chunk in chunks:
        print(f"  ({len(chunk):>3}) {chunk!r}")
    print()


def main() -> None:
    # Tenta separar nesta ordem e só desce quando o pedaço ainda é grande demais.
    print(f"separadores: {['\n\n', '\n', ' ', '']}\n")
    show(200, 0)  # cabem parágrafos
    show(80, 0)  # precisa quebrar linhas e palavras
    show(80, 30)  # a sobreposição repete o fim do pedaço anterior


if __name__ == "__main__":
    main()
