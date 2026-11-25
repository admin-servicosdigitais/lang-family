"""Passo 5: a avaliação do laboratório (``labs/rag/evaluate.py``).

O laboratório compara Chroma, Qdrant e Milvus. Aqui há um banco em memória, para o
arquivo rodar sozinho; ele e o ``main`` são só para a aula. Usa o modelo local.
"""

import hashlib
import json
import math
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# --- Da aula 05: fragmentação -------------------------------------------------------

# Na aula, o caminho aponta para os dados do laboratório.
DATA_DIR = Path(__file__).parents[2] / "labs" / "rag" / "data"


# --- Do laboratório: rag/chunking.py ------------------------------------------------


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


# --- Do laboratório: rag/embeddings.py ----------------------------------------------


Vector = list[float]


class Embedder(Protocol):
    dimension: int

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]: ...

    def embed_query(self, text: str) -> Vector: ...


def normalize(vector: Sequence[float]) -> Vector:
    norm = math.sqrt(sum(value * value for value in vector))
    return [float(value) / norm for value in vector] if norm else [float(v) for v in vector]


class HashEmbedder:
    """Saco de palavras com hashing. Não entende sinônimos: serve para testar a
    mecânica dos bancos vetoriais sem baixar modelo nenhum."""

    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension

    def embed_query(self, text: str) -> Vector:
        vector = [0.0] * self.dimension
        for word in re.findall(r"\w+", text.lower()):
            digest = hashlib.sha256(word.encode()).digest()
            vector[int.from_bytes(digest[:4], "big") % self.dimension] += 1.0
        return normalize(vector)

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        return [self.embed_query(text) for text in texts]


class FastEmbedEmbedder:
    """Modelo multilíngue executado localmente via ONNX, sem chave de API."""

    MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    dimension = 384

    def __init__(self) -> None:
        from fastembed import TextEmbedding

        default_cache = Path(__file__).parents[2] / ".cache" / "fastembed"
        cache_dir = os.getenv("FASTEMBED_CACHE_PATH", str(default_cache))
        self._model = TextEmbedding(self.MODEL, cache_dir=cache_dir)

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        return [normalize(vector.tolist()) for vector in self._model.embed(list(texts))]

    def embed_query(self, text: str) -> Vector:
        return self.embed_documents([text])[0]


# --- Do laboratório: rag/stores/base.py ---------------------------------------------


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class VectorStore(Protocol):
    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None: ...

    def query(self, vector: Vector, k: int, *, category: str | None = None) -> list[Hit]: ...

    def delete_source(self, source: str) -> None: ...

    def count(self) -> int: ...

    def close(self) -> None:
        """Libera conexões; não apaga os dados."""
        ...

    def drop(self) -> None:
        """Apaga a coleção/namespace usada por esta instância."""
        ...


# --- Do laboratório: rag/evaluate.py ------------------------------------------------


@dataclass(frozen=True)
class Case:
    question: str
    source: str
    section: str


@dataclass(frozen=True)
class Metrics:
    recall_at_k: float
    mrr: float


def load_golden(path: Path = DATA_DIR / "golden.json") -> list[Case]:
    return [Case(**item) for item in json.loads(path.read_text(encoding="utf-8"))]


def evaluate(
    store: VectorStore, embedder: Embedder, cases: Sequence[Case], *, k: int = 3
) -> Metrics:
    if not cases:
        raise ValueError("informe ao menos um caso de avaliação")
    found = 0
    reciprocal_ranks = 0.0
    for case in cases:
        hits = store.query(embedder.embed_query(case.question), k)
        for position, hit in enumerate(hits, start=1):
            if (hit.chunk.source, hit.chunk.section) == (case.source, case.section):
                found += 1
                reciprocal_ranks += 1 / position
                break
    return Metrics(found / len(cases), reciprocal_ranks / len(cases))


# --- Somente da aula ----------------------------------------------------------------


class MemoryStore:
    def __init__(self) -> None:
        self.rows: dict[str, tuple[Chunk, Vector]] = {}

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None:
        self.rows.update({c.id: (c, v) for c, v in zip(chunks, vectors, strict=True)})

    def query(self, vector: Vector, k: int, *, category: str | None = None) -> list[Hit]:
        hits = [
            Hit(chunk, sum(x * y for x, y in zip(vector, stored, strict=True)))
            for chunk, stored in self.rows.values()
            if category is None or chunk.category == category
        ]
        return sorted(hits, key=lambda hit: -hit.score)[:k]

    def delete_source(self, source: str) -> None:
        self.rows = {i: row for i, row in self.rows.items() if row[0].source != source}

    def count(self) -> int:
        return len(self.rows)

    def close(self) -> None:
        pass

    def drop(self) -> None:
        self.rows.clear()


def main() -> None:
    chunks = load_corpus()
    cases = load_golden()
    embedders: dict[str, Embedder] = {
        "hash (léxico)": HashEmbedder(384),
        "MiniLM (semântico)": FastEmbedEmbedder(),
    }
    print(f"{len(chunks)} trechos, {len(cases)} perguntas\n")
    print(f"{'embeddings':<20} {'k':>2} {'recall@k':>8} {'MRR':>6}")
    for label, embedder in embedders.items():
        store = MemoryStore()
        store.upsert(chunks, embedder.embed_documents([c.contextual_text for c in chunks]))
        for k in (1, 3):
            metrics = evaluate(store, embedder, cases, k=k)
            print(f"{label:<20} {k:>2} {metrics.recall_at_k:>8.2f} {metrics.mrr:>6.2f}")


if __name__ == "__main__":
    main()
