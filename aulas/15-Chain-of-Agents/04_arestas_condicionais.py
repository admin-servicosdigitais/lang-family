"""Passo 4: um laço com aresta condicional, e o limite de recursão."""

from typing import Any, TypedDict

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph


class State(TypedDict, total=False):
    chunks: list[str]
    index: int
    notes: str


def work(state: State) -> State:
    index = state.get("index", 0)
    notes = state.get("notes", "") + state["chunks"][index]
    return {"notes": notes, "index": index + 1}


def next_step(state: State) -> str:
    """Decide para onde ir a partir do estado."""
    return "worker" if state.get("index", 0) < len(state["chunks"]) else END


def main() -> None:
    graph = StateGraph(State)
    graph.add_node("worker", work)
    graph.add_conditional_edges(START, next_step, ["worker", END])
    graph.add_conditional_edges("worker", next_step, ["worker", END])
    app = graph.compile()

    for chunks in (["a", "b", "c"], []):
        result: Any = app.invoke({"chunks": chunks})
        print(f"trechos={chunks} notas={result.get('notes', '')!r}")

    try:
        app.invoke({"chunks": ["x"] * 10}, {"recursion_limit": 5})
    except GraphRecursionError:
        print("10 trechos com recursion_limit=5: o grafo é interrompido")


if __name__ == "__main__":
    main()
