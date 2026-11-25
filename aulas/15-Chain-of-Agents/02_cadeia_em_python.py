"""Passo 2: a ideia do Chain-of-Agents em Python puro, sem framework."""

import re

CHUNKS = [
    "O projeto Aurora é liderado por Marina.",
    "Relatório administrativo sem relação com o assunto.",
    "Marina trabalha no escritório de Recife.",
]


def worker(notes: str, chunk: str) -> str:
    """Faz o papel do modelo: vê as anotações e UM trecho, e devolve as anotações."""
    if found := re.search(r"liderado por (\w+)", chunk):
        notes += f"lider={found[1]};"
    leader = re.search(r"lider=(\w+)", notes)
    if leader and (found := re.search(rf"{leader[1]} trabalha no escritório de (\w+)", chunk)):
        notes += f"cidade={found[1]};"
    return notes


def manager(notes: str) -> str:
    """Vê somente as anotações finais."""
    city = re.search(r"cidade=(\w+)", notes)
    return city[1] if city else "informação não encontrada"


def run(chunks: list[str]) -> str:
    notes = ""
    for number, chunk in enumerate(chunks, start=1):
        notes = worker(notes, chunk)
        print(f"  trabalhador {number}: {notes!r}")
    return manager(notes)


def main() -> None:
    print("todos os trechos")
    print(f"  gerente: {run(CHUNKS)}")
    print("sem o primeiro elo")
    print(f"  gerente: {run(CHUNKS[1:])}")
    print("ordem invertida")
    print(f"  gerente: {run(CHUNKS[::-1])}")


if __name__ == "__main__":
    main()
