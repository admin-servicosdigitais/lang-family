"""Carga e fragmentação dos documentos: primeiro por seção, depois por tamanho."""

from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

DATA_DIR = Path(__file__).parent / "data"


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
    if not raw.startswith("---\n"):
        return {}, raw
    header, body = raw[4:].split("\n---\n", 1)
    pairs = (line.split(":", 1) for line in header.splitlines() if ":" in line)
    return {key.strip(): value.strip() for key, value in pairs}, body


def chunk_document(
    source: str, raw: str, *, chunk_size: int = 500, chunk_overlap: int = 50
) -> list[Chunk]:
    metadata, body = _front_matter(raw)
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
