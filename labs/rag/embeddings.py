"""Geradores de embeddings: um semântico (local) e um determinístico (para testes)."""

import hashlib
import math
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

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
