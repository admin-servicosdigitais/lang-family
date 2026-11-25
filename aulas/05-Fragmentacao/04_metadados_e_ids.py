"""Passo 4: cabeçalho do arquivo, ids estáveis e reingestão sem sobras."""

import re
from dataclasses import dataclass

from langchain_text_splitters import MarkdownHeaderTextSplitter

# --- Novo neste passo -------------------------------------------------------------

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
        return f"{self.section}\n{self.text}"


def front_matter(raw: str) -> tuple[dict[str, str], str]:
    match = _FRONT_MATTER.match(raw)
    if match is None:
        return {}, raw
    pairs = (line.split(":", 1) for line in match[1].splitlines() if ":" in line)
    return {key.strip(): value.strip() for key, value in pairs}, raw[match.end() :]


# --- Visto no passo 3 (só a divisão por seção) ------------------------------------


def chunk_document(source: str, raw: str) -> list[Chunk]:
    metadata, body = front_matter(raw)
    splitter = MarkdownHeaderTextSplitter([("#", "title"), ("##", "section")])
    return [
        Chunk(
            id=f"{source}#{index}",
            text=piece.page_content,
            source=source,
            section=piece.metadata.get("section", ""),
            category=metadata.get("category", "geral"),
        )
        for index, piece in enumerate(splitter.split_text(body))
    ]


# --- Novo neste passo: reingerir --------------------------------------------------


def upsert(store: dict[str, Chunk], chunks: list[Chunk]) -> None:
    store.update({chunk.id: chunk for chunk in chunks})


def replace_source(store: dict[str, Chunk], chunks: list[Chunk]) -> None:
    sources = {chunk.source for chunk in chunks}
    for chunk_id in [i for i, c in store.items() if c.source in sources]:
        del store[chunk_id]
    upsert(store, chunks)


V1 = "---\ncategory: financeiro\n---\n# P\n## Pix\nPix tem desconto.\n## Boleto\nBoleto aceito.\n"
V2 = "---\ncategory: financeiro\n---\n# P\n## Pix\nSó Pix agora.\n"


def main() -> None:
    chunks = chunk_document("pagamentos.md", V1)
    for chunk in chunks:
        print(f"{chunk.id} [{chunk.category}] embedding de: {chunk.contextual_text!r}")

    store: dict[str, Chunk] = {}
    upsert(store, chunks)
    upsert(store, chunks)
    print(f"\nreingestão igual: {len(store)} trechos (ids estáveis não duplicam)")

    upsert(store, chunk_document("pagamentos.md", V2))
    print(f"documento encolheu, só upsert : {[c.text for c in store.values()]}")

    replace_source(store, chunk_document("pagamentos.md", V2))
    print(f"substituindo o documento      : {[c.text for c in store.values()]}")


if __name__ == "__main__":
    main()
