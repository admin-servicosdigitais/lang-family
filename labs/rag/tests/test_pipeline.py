"""Comportamento do pipeline RAG, com embeddings e modelo determinísticos."""

import uuid

import pytest
from langchain_core.messages import AIMessage

from labs.rag.chunking import chunk_document, load_corpus
from labs.rag.embeddings import HashEmbedder
from labs.rag.pipeline import NO_ANSWER, RagPipeline
from labs.rag.stores.qdrant import QdrantStore
from labs.shared.fake_model import ScriptedChatModel


def build(llm: ScriptedChatModel, *, min_score: float = 0.2) -> RagPipeline:
    store = QdrantStore(f"t{uuid.uuid4().hex}", 512)
    pipeline = RagPipeline(store, HashEmbedder(512), llm, min_score=min_score)
    pipeline.ingest(load_corpus())
    return pipeline


def test_chunking_keeps_section_and_category() -> None:
    chunks = load_corpus()

    pix = next(c for c in chunks if c.section == "Desconto no Pix")
    assert (pix.source, pix.category) == ("pagamentos.md", "financeiro")
    assert "5%" in pix.text
    assert "category:" not in " ".join(c.text for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)


def test_long_sections_are_split_within_size_limit() -> None:
    text = "# T\n\n## Longa\n" + " ".join(f"palavra{i}" for i in range(400))

    chunks = chunk_document("x.md", text, chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(len(c.text) <= 200 and c.section == "Longa" for c in chunks)


@pytest.mark.parametrize(
    "raw",
    ["---\n\nTexto depois de uma régua.", "---\ncategory: x\n# T\nTexto sem fechamento."],
    ids=["regua-horizontal", "sem-fechamento"],
)
def test_unclosed_front_matter_is_treated_as_content(raw: str) -> None:
    chunks = chunk_document("x.md", raw)

    assert chunks
    assert {c.category for c in chunks} == {"geral"}


def test_front_matter_closing_at_end_of_file() -> None:
    assert chunk_document("x.md", "---\ncategory: x\n---") == []


def test_front_matter_with_windows_line_endings() -> None:
    chunks = chunk_document("x.md", "---\r\ncategory: x\r\n---\r\n# T\r\nTexto")

    assert [(c.category, c.text) for c in chunks] == [("x", "Texto")]


def test_reingesting_does_not_duplicate() -> None:
    pipeline = build(ScriptedChatModel.from_script([]))
    before = pipeline.store.count()

    pipeline.ingest(load_corpus())

    assert pipeline.store.count() == before


def test_shrunk_document_leaves_no_orphan_chunks() -> None:
    pipeline = build(ScriptedChatModel.from_script([]))
    before = pipeline.store.count()
    old = [c for c in load_corpus() if c.source == "pagamentos.md"]

    pipeline.ingest(chunk_document("pagamentos.md", "# P\n\n## Única\nSó Pix agora."))

    assert pipeline.store.count() == before - len(old) + 1


def test_answer_sends_context_and_maps_citations() -> None:
    llm = ScriptedChatModel.from_script([AIMessage("Há 5% de desconto [1]. Ver [99].")])
    pipeline = build(llm)

    answer = pipeline.answer("Pagamentos via Pix recebem desconto?")

    system, human = llm.calls[0]
    assert "SOMENTE o contexto" in system.text
    assert "Pagamentos via Pix recebem 5% de desconto" in human.text
    assert [c.section for c in answer.citations] == ["Desconto no Pix"]


def test_category_filter_limits_retrieval() -> None:
    pipeline = build(ScriptedChatModel.from_script([]), min_score=-1)

    hits = pipeline.retrieve("prazo", category="juridico")

    assert hits
    assert {h.chunk.source for h in hits} == {"privacidade.md"}


@pytest.mark.parametrize("question", ["xyzzy plugh", ""])
def test_irrelevant_question_does_not_call_the_model(question: str) -> None:
    llm = ScriptedChatModel.from_script([])
    pipeline = build(llm, min_score=0.5)

    answer = pipeline.answer(question)

    assert answer.text == NO_ANSWER
    assert answer.citations == ()
    assert llm.calls == []
