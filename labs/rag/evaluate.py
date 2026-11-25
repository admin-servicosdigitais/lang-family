"""Avaliação da recuperação (recall@k e MRR) sobre o conjunto de referência.

Execute: ``uv run python -m labs.rag.evaluate``
"""

import json
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from labs.rag.chunking import DATA_DIR, load_corpus
from labs.rag.embeddings import Embedder, FastEmbedEmbedder, HashEmbedder
from labs.rag.stores.base import VectorStore
from labs.rag.stores.chroma import ChromaStore
from labs.rag.stores.milvus import MilvusStore
from labs.rag.stores.qdrant import QdrantStore


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


def main() -> None:
    chunks = load_corpus()
    cases = load_golden()
    embedders: dict[str, Embedder] = {
        "hash (léxico)": HashEmbedder(384),
        "MiniLM (semântico)": FastEmbedEmbedder(),
    }
    print(f"{len(chunks)} trechos, {len(cases)} perguntas\n")
    print(f"{'banco':<8} {'embeddings':<20} {'recall@3':>8} {'MRR':>6}")
    with tempfile.TemporaryDirectory() as tmp:
        for number, (label, embedder) in enumerate(embedders.items()):
            vectors = embedder.embed_documents([c.contextual_text for c in chunks])
            name = f"eval_{number}"
            stores: dict[str, VectorStore] = {
                "chroma": ChromaStore(name),
                "qdrant": QdrantStore(name, embedder.dimension),
                "milvus": MilvusStore(name, embedder.dimension, uri=f"{tmp}/{name}.db"),
            }
            for store_name, store in stores.items():
                store.upsert(chunks, vectors)
                metrics = evaluate(store, embedder, cases)
                print(
                    f"{store_name:<8} {label:<20} {metrics.recall_at_k:>8.2f} {metrics.mrr:>6.2f}"
                )
                store.close()


if __name__ == "__main__":
    main()
