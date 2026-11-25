"""Passo 5: os geradores de embeddings do laboratório (``labs/rag/embeddings.py``).

Só o ``main`` é da aula. Na primeira execução são baixados cerca de 220 MB.
"""

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


# --- Somente da aula: demonstração --------------------------------------------------


def best_match(embedder: Embedder, question: str, documents: list[str]) -> tuple[float, str]:
    """Código de negócio: depende do contrato, não de um modelo específico."""
    query = embedder.embed_query(question)
    scored = [
        (sum(x * y for x, y in zip(query, vector, strict=True)), text)
        for text, vector in zip(documents, embedder.embed_documents(documents), strict=True)
    ]
    return max(scored)


def main() -> None:
    documents = [
        "Pedidos acima de R$ 299 têm frete grátis.",
        "A garantia cobre defeitos de fabricação por um ano.",
        "A entrega no norte leva até 12 dias.",
    ]
    question = "O envio é de graça?"
    print(f"pergunta: {question}")
    embedders: list[Embedder] = [HashEmbedder(), FastEmbedEmbedder()]
    for embedder in embedders:
        score, text = best_match(embedder, question, documents)
        print(f"  {type(embedder).__name__:<18} {embedder.dimension:>3}d  {score:.2f}  {text}")


if __name__ == "__main__":
    main()
