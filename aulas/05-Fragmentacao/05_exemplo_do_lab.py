"""Passo 5: a fragmentação do laboratório (``labs/rag/chunking.py``).

Só mudam o caminho dos dados e o ``main``, que é da aula.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# Na aula, o caminho aponta para os documentos do laboratório.
DATA_DIR = Path(__file__).parents[2] / "labs" / "rag" / "data"
# Cabeçalho entre duas linhas "---" no início do arquivo.
_FRONT_MATTER = re.compile(r"\A---\n(.*?)^---(?:\n|\Z)", re.DOTALL | re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    section: str
    category: str

    @property
    def contextual_text(self) -> str:
        """Texto usado no embedding: o título da seção dá contexto ao trecho."""
        return f"{self.section}\n{self.text}"


def _front_matter(raw: str) -> tuple[dict[str, str], str]:
    match = _FRONT_MATTER.match(raw)
    if match is None:
        # Sem fechamento, o "---" é só uma régua horizontal do Markdown.
        return {}, raw
    pairs = (line.split(":", 1) for line in match[1].splitlines() if ":" in line)
    return {key.strip(): value.strip() for key, value in pairs}, raw[match.end() :]


def chunk_document(
    source: str, raw: str, *, chunk_size: int = 500, chunk_overlap: int = 50
) -> list[Chunk]:
    metadata, body = _front_matter(raw.replace("\r\n", "\n"))
    by_section = MarkdownHeaderTextSplitter([("#", "title"), ("##", "section")])
    by_size = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    pieces = by_size.split_documents(by_section.split_text(body))
    return [
        # O id depende só da origem e da posição: reingerir sobrescreve, não duplica.
        Chunk(
            id=f"{source}#{index}",
            text=piece.page_content,
            source=source,
            section=piece.metadata.get("section") or piece.metadata.get("title", ""),
            category=metadata.get("category", "geral"),
        )
        for index, piece in enumerate(pieces)
    ]


def load_corpus(directory: Path = DATA_DIR) -> list[Chunk]:
    return [
        chunk
        for path in sorted(directory.glob("*.md"))
        for chunk in chunk_document(path.name, path.read_text(encoding="utf-8"))
    ]


# --- Somente da aula: demonstração --------------------------------------------------


def main() -> None:
    chunks = load_corpus()
    print(f"{len(chunks)} trechos, o maior com {max(len(c.text) for c in chunks)} caracteres\n")
    for chunk in chunks:
        print(f"{chunk.id:<18} {chunk.category:<12} {chunk.section}")
    print(f"\nTexto usado no embedding de {chunks[0].id}:\n{chunks[0].contextual_text}")


if __name__ == "__main__":
    main()
