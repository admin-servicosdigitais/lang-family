"""Fixtures dos bancos vetoriais: o mesmo contrato roda em todos eles."""

import time
import uuid
import warnings
from collections.abc import Callable, Iterator
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
from pinecone import Pinecone

from labs.rag.stores.base import VectorStore
from labs.rag.stores.chroma import ChromaStore
from labs.rag.stores.milvus import MilvusStore
from labs.rag.stores.pinecone_store import PineconeStore
from labs.rag.stores.qdrant import QdrantStore
from labs.rag.stores.weaviate_store import WeaviateStore
from labs.rag.tests.fake_pinecone import FakePinecone
from labs.shared.fake_http import fake_server
from labs.shared.settings import require_env

DIMENSION = 8
QDRANT_URL = "http://localhost:16333"
WEAVIATE_READY_URL = "http://localhost:18080/v1/.well-known/ready"


@dataclass
class Backend:
    store: VectorStore
    # Tempo para a escrita ficar visível (só o serviço em nuvem precisa).
    settle_timeout: float = 0.0

    def eventually(self, check: Callable[[], None]) -> None:
        deadline = time.monotonic() + self.settle_timeout
        while True:
            try:
                return check()
            except AssertionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(1)


def _require_service(url: str) -> None:
    try:
        httpx.get(url, timeout=1).raise_for_status()
    except httpx.HTTPError:
        pytest.skip(f"serviço indisponível em {url} (suba infra/vectordbs.compose.yml)")


def _open(kind: str, name: str, tmp_path: Path) -> Iterator[Backend]:
    if kind == "chroma":
        yield Backend(ChromaStore(name))
    elif kind == "qdrant-local":
        yield Backend(QdrantStore(name, DIMENSION))
    elif kind == "qdrant-server":
        _require_service(f"{QDRANT_URL}/readyz")
        yield Backend(QdrantStore(name, DIMENSION, location=QDRANT_URL))
    elif kind == "milvus-lite":
        yield Backend(MilvusStore(name, DIMENSION, uri=str(tmp_path / "milvus.db")))
    elif kind == "weaviate":
        _require_service(WEAVIATE_READY_URL)
        yield Backend(WeaviateStore(name))
    elif kind == "pinecone-fake":
        with fake_server(FakePinecone().handle) as url:
            index = Pinecone(api_key="teste").index(host=url)
            yield Backend(PineconeStore(index, name))
    elif kind == "pinecone-cloud":
        key = require_env("PINECONE_API_KEY")
        store = PineconeStore.from_cloud(
            name, DIMENSION, api_key=key, index_name="lang-family-contract"
        )
        yield Backend(store, settle_timeout=60)
    else:
        raise ValueError(kind)


@pytest.fixture(
    params=[
        "chroma",
        "qdrant-local",
        pytest.param("qdrant-server", marks=pytest.mark.docker),
        "milvus-lite",
        pytest.param("weaviate", marks=pytest.mark.docker),
        "pinecone-fake",
        pytest.param("pinecone-cloud", marks=pytest.mark.live),
    ]
)
def backend(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Backend]:
    name = f"test_{uuid.uuid4().hex[:12]}"
    for backend_ in _open(request.param, name, tmp_path):
        try:
            yield backend_
        finally:
            # A limpeza não pode esconder a falha do teste.
            with suppress(Exception):
                backend_.store.drop()
            try:
                backend_.store.close()
            except Exception as error:
                warnings.warn(f"falha ao fechar o banco: {error}", stacklevel=1)
