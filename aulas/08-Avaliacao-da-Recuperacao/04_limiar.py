"""Passo 4: o efeito do corte de pontuação, na base do laboratório.

Usa o modelo local de embeddings (aula 06): sem chave; baixa 220 MB na primeira vez.
"""

import json
from pathlib import Path

from fastembed import TextEmbedding
from langchain_text_splitters import MarkdownHeaderTextSplitter

ROOT = Path(__file__).parents[2]
DATA = ROOT / "labs" / "rag" / "data"
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OFF_TOPIC = ["Qual a previsão do tempo amanhã?", "Quem ganhou a copa?", "Receita de bolo"]


def sections() -> list[tuple[str, str, str]]:
    """(arquivo, seção, texto do embedding), como na aula 05."""
    splitter = MarkdownHeaderTextSplitter([("#", "title"), ("##", "section")])
    result = []
    for path in sorted(DATA.glob("*.md")):
        body = path.read_text(encoding="utf-8").split("---\n", 2)[2]
        for piece in splitter.split_text(body):
            section = piece.metadata["section"]
            result.append((path.name, section, f"{section}\n{piece.page_content}"))
    return result


def main() -> None:
    model = TextEmbedding(MODEL, cache_dir=str(ROOT / ".cache" / "fastembed"))

    def embed(texts: list[str]) -> list[list[float]]:
        vectors = [vector.tolist() for vector in model.embed(texts)]
        return [[x / sum(y * y for y in v) ** 0.5 for x in v] for v in vectors]

    corpus = sections()
    vectors = embed([text for _, _, text in corpus])

    def search(question: str, k: int = 4) -> list[tuple[float, str, str]]:
        query = embed([question])[0]
        scored = [
            (sum(a * b for a, b in zip(query, v, strict=True)), source, section)
            for (source, section, _), v in zip(corpus, vectors, strict=True)
        ]
        return sorted(scored, reverse=True)[:k]

    golden = json.loads((DATA / "golden.json").read_text(encoding="utf-8"))
    results = [(case, search(case["question"])) for case in golden]
    off_topic = [search(question)[0][0] for question in OFF_TOPIC]

    print("corte  trecho certo entregue  fora do tema que passa")
    for threshold in (0.0, 0.2, 0.25, 0.3, 0.4, 0.5):
        delivered = sum(
            any(
                (source, section) == (case["source"], case["section"]) and score >= threshold
                for score, source, section in hits
            )
            for case, hits in results
        )
        leaked = sum(score >= threshold for score in off_topic)
        print(
            f"{threshold:>5}  {delivered:>2} de {len(golden)}{'':<14}{leaked} de {len(OFF_TOPIC)}"
        )


if __name__ == "__main__":
    main()
