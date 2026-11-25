"""Contrato comum, verificado com vetores escolhidos à mão (cosseno conhecido)."""

import pytest

from labs.rag.chunking import Chunk
from labs.rag.tests.conftest import Backend

# Ângulos conhecidos em relação à consulta [1, 0, ...]: cos = 1.0, 0.8, 0.6 e 0.0.
VECTORS = {
    "a.md#0": [1.0, 0.0, 0, 0, 0, 0, 0, 0],
    "a.md#1": [0.8, 0.6, 0, 0, 0, 0, 0, 0],
    "b.md#0": [0.6, 0.8, 0, 0, 0, 0, 0, 0],
    "b.md#1": [0.0, 1.0, 0, 0, 0, 0, 0, 0],
}
QUERY = [1.0, 0, 0, 0, 0, 0, 0, 0]


def make_chunk(chunk_id: str, text: str = "", category: str | None = None) -> Chunk:
    source = chunk_id.split("#")[0]
    return Chunk(
        id=chunk_id,
        text=text or f"conteúdo de {chunk_id}",
        source=source,
        section="Seção ç/ã",
        category=category or ("logistica" if source == "a.md" else "financeiro"),
    )


@pytest.fixture
def filled(backend: Backend) -> Backend:
    backend.store.upsert([make_chunk(i) for i in VECTORS], list(VECTORS.values()))

    def check() -> None:
        assert backend.store.count() == 4

    backend.eventually(check)
    return backend


def test_query_ranks_by_cosine_similarity(filled: Backend) -> None:
    def check() -> None:
        hits = filled.store.query(QUERY, 3)
        assert [h.chunk.id for h in hits] == ["a.md#0", "a.md#1", "b.md#0"]
        # Mesma escala em todos os bancos, apesar de uns devolverem distância.
        assert [h.score for h in hits] == pytest.approx([1.0, 0.8, 0.6], abs=1e-3)

    filled.eventually(check)


def test_hit_round_trips_the_whole_chunk(filled: Backend) -> None:
    def check() -> None:
        assert filled.store.query(QUERY, 1)[0].chunk == make_chunk("a.md#0")

    filled.eventually(check)


def test_metadata_filter(filled: Backend) -> None:
    def check() -> None:
        hits = filled.store.query(QUERY, 4, category="financeiro")
        assert [h.chunk.id for h in hits] == ["b.md#0", "b.md#1"]

    filled.eventually(check)


def test_filter_value_is_not_interpreted_as_expression(filled: Backend) -> None:
    hostile = 'x" or category != "x'

    assert filled.store.query(QUERY, 4, category=hostile) == []


def test_upsert_with_same_id_replaces_record(filled: Backend) -> None:
    updated = make_chunk("b.md#1", text="texto novo")

    filled.store.upsert([updated], [QUERY])

    def check() -> None:
        assert filled.store.count() == 4
        best = filled.store.query(QUERY, 2)
        assert updated in [h.chunk for h in best]
        assert [h.score for h in best] == pytest.approx([1.0, 1.0], abs=1e-3)

    filled.eventually(check)


def test_upsert_of_nothing_is_a_no_op(filled: Backend) -> None:
    filled.store.upsert([], [])

    def check() -> None:
        assert filled.store.count() == 4

    filled.eventually(check)


def test_delete_by_source(filled: Backend) -> None:
    filled.store.delete_source("a.md")

    def check() -> None:
        assert filled.store.count() == 2
        assert {h.chunk.source for h in filled.store.query(QUERY, 4)} == {"b.md"}

    filled.eventually(check)
