"""RAG de ponta a ponta com um LLM real: ``uv run pytest --live labs/rag``."""

import pytest

from labs.rag.chunking import load_corpus
from labs.rag.embeddings import FastEmbedEmbedder
from labs.rag.pipeline import NO_ANSWER, RagPipeline
from labs.rag.stores.qdrant import QdrantStore
from labs.shared.llm import live_chat_model

pytestmark = [pytest.mark.live, pytest.mark.embeddings]


@pytest.fixture(scope="module")
def pipeline() -> RagPipeline:
    embedder = FastEmbedEmbedder()
    store = QdrantStore("live", embedder.dimension)
    rag = RagPipeline(store, embedder, live_chat_model())
    rag.ingest(load_corpus())
    return rag


def test_grounded_answer_with_citation(pipeline: RagPipeline) -> None:
    answer = pipeline.answer("Qual o desconto para quem paga com Pix?")

    assert "5%" in answer.text
    assert "pagamentos.md" in {c.source for c in answer.citations}


def test_admits_when_context_has_no_answer(pipeline: RagPipeline) -> None:
    answer = pipeline.answer("A loja vende passagens aéreas para o Japão?")

    assert NO_ANSWER.rstrip(".") in answer.text
