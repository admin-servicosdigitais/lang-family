"""Qualidade da recuperação com um modelo de embeddings de verdade (local)."""

from pathlib import Path

import pytest

from labs.rag.chunking import load_corpus
from labs.rag.embeddings import Embedder, FastEmbedEmbedder, HashEmbedder
from labs.rag.evaluate import Metrics, evaluate, load_golden
from labs.rag.stores.base import VectorStore
from labs.rag.stores.chroma import ChromaStore
from labs.rag.stores.milvus import MilvusStore
from labs.rag.stores.qdrant import QdrantStore

pytestmark = pytest.mark.embeddings


@pytest.fixture(scope="module")
def semantic() -> FastEmbedEmbedder:
    return FastEmbedEmbedder()


def metrics_for(embedder: Embedder, name: str) -> Metrics:
    chunks = load_corpus()
    store = QdrantStore(name, embedder.dimension)
    store.upsert(chunks, embedder.embed_documents([c.contextual_text for c in chunks]))
    return evaluate(store, embedder, load_golden(), k=3)


def test_semantic_retrieval_meets_quality_bar(semantic: FastEmbedEmbedder) -> None:
    metrics = metrics_for(semantic, "semantic")

    assert metrics.recall_at_k >= 0.9
    assert metrics.mrr >= 0.75


def test_semantic_beats_lexical_baseline(semantic: FastEmbedEmbedder) -> None:
    lexical = metrics_for(HashEmbedder(384), "lexical")

    assert metrics_for(semantic, "semantic2").mrr > lexical.mrr


def test_databases_agree_on_ranking(semantic: FastEmbedEmbedder, tmp_path: Path) -> None:
    chunks = load_corpus()
    vectors = semantic.embed_documents([c.contextual_text for c in chunks])
    stores: list[VectorStore] = [
        ChromaStore("acordo"),
        QdrantStore("acordo", semantic.dimension),
        MilvusStore("acordo", semantic.dimension, uri=str(tmp_path / "m.db")),
    ]
    query = semantic.embed_query("Tem abatimento se eu pagar com Pix?")
    rankings = []
    for store in stores:
        store.upsert(chunks, vectors)
        rankings.append([h.chunk.id for h in store.query(query, 3)])
        store.drop()
        store.close()

    assert rankings[0] == rankings[1] == rankings[2]
