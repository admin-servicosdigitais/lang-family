"""Passo 3: dividir primeiro pela estrutura do documento, depois pelo tamanho."""

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

DOCUMENT = """# Trocas e devoluções

## Prazo para desistência
Compras podem ser canceladas em até 7 dias.

O estorno é integral.

## Troca por defeito
Produtos com defeito podem ser trocados em até 90 dias após a entrega.
É preciso enviar fotos do problema pelo chat antes de despachar o item.

### Exceções
Itens de higiene abertos não podem ser devolvidos.
"""


def main() -> None:
    by_section = MarkdownHeaderTextSplitter([("#", "title"), ("##", "section")])
    sections = by_section.split_text(DOCUMENT)
    print("1. por seção")
    for piece in sections:
        print(f"  {piece.metadata}")
        print(f"    {piece.page_content!r}")

    by_size = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=10)
    print("\n2. depois por tamanho (os metadados acompanham cada pedaço)")
    for piece in by_size.split_documents(sections):
        print(f"  [{piece.metadata['section']}] {piece.page_content!r}")


if __name__ == "__main__":
    main()
