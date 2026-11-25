"""Passo 4: ingerir substituindo o documento inteiro, na ordem segura."""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    text: str


Embed = Callable[[list[str]], list[list[float]]]
Store = dict[str, tuple[Chunk, list[float]]]


def delete_source(store: Store, source: str) -> None:
    for chunk_id in [i for i, (chunk, _) in store.items() if chunk.source == source]:
        del store[chunk_id]


def ingest(store: Store, embed: Embed, chunks: list[Chunk]) -> int:
    if not chunks:
        return 0
    # 1. Calcular os vetores primeiro: se falhar, nada foi apagado.
    vectors = embed([chunk.text for chunk in chunks])
    # 2. Apagar os documentos que vão ser substituídos (evita trechos órfãos).
    for source in sorted({chunk.source for chunk in chunks}):
        delete_source(store, source)
    # 3. Gravar.
    store.update({c.id: (c, v) for c, v in zip(chunks, vectors, strict=True)})
    return len(chunks)


def works(texts: list[str]) -> list[list[float]]:
    return [[float(len(text))] for text in texts]


def broken(texts: list[str]) -> list[list[float]]:
    raise ConnectionError("serviço de embeddings fora do ar")


def show(label: str, store: Store) -> None:
    print(f"{label:<28} {[chunk.text for chunk, _ in store.values()]}")


def main() -> None:
    store: Store = {}
    old = [Chunk("p#0", "pagamentos", "Pix e boleto."), Chunk("p#1", "pagamentos", "Cartão.")]
    ingest(store, works, old)
    show("primeira carga:", store)

    try:
        ingest(store, broken, [Chunk("p#0", "pagamentos", "Só Pix.")])
    except ConnectionError as error:
        show(f"falha ({error.__class__.__name__}):", store)

    ingest(store, works, [Chunk("p#0", "pagamentos", "Só Pix.")])
    show("documento atualizado:", store)

    ingest(store, works, [])
    show("documento esvaziado:", store)
    print("Com zero trechos não há como saber o que apagar: o conteúdo antigo fica.")


if __name__ == "__main__":
    main()
