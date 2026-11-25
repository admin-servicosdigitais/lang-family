"""Os bancos embarcados mantêm os dados depois de reabertos."""

from collections.abc import Callable
from pathlib import Path

import pytest

from labs.rag.chunking import Chunk
from labs.rag.stores.base import VectorStore
from labs.rag.stores.chroma import ChromaStore
from labs.rag.stores.milvus import MilvusStore

OPENERS: dict[str, Callable[[Path], VectorStore]] = {
    "chroma": lambda path: ChromaStore("persistencia", path=str(path / "chroma")),
    "milvus-lite": lambda path: MilvusStore("persistencia", 4, uri=str(path / "milvus.db")),
}


@pytest.mark.parametrize("kind", sorted(OPENERS))
def test_data_survives_reopening(kind: str, tmp_path: Path) -> None:
    chunk = Chunk("a.md#0", "texto persistido", "a.md", "S", "geral")
    first = OPENERS[kind](tmp_path)
    first.upsert([chunk], [[1.0, 0.0, 0.0, 0.0]])
    first.close()

    reopened = OPENERS[kind](tmp_path)

    assert reopened.count() == 1
    assert reopened.query([1.0, 0.0, 0.0, 0.0], 1)[0].chunk == chunk
    reopened.close()
