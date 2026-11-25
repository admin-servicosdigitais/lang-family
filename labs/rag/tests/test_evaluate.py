"""A métrica também é testada."""

import pytest

from labs.rag.embeddings import HashEmbedder
from labs.rag.evaluate import evaluate
from labs.rag.stores.qdrant import QdrantStore


def test_evaluation_without_cases_is_rejected() -> None:
    with pytest.raises(ValueError, match="ao menos um caso"):
        evaluate(QdrantStore("sem_casos", 8), HashEmbedder(8), [])
