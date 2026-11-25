"""Passo 3: os elementos de um grafo no LangGraph: estado, nós, arestas e redutores."""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict, total=False):
    chunk: str
    notes: str  # sem redutor: o valor novo substitui o antigo
    history: Annotated[list[str], operator.add]  # com redutor: o novo é somado


def read(state: State) -> State:
    # O nó devolve só o que mudou.
    notes = f"li: {state['chunk']}"
    return {"notes": notes, "history": [notes]}


def finish(state: State) -> State:
    return {"notes": "pronto", "history": ["pronto"]}


def main() -> None:
    graph = StateGraph(State)
    graph.add_node("read", read)
    graph.add_node("finish", finish)
    graph.add_edge(START, "read")
    graph.add_edge("read", "finish")
    graph.add_edge("finish", END)
    app = graph.compile()

    result = app.invoke({"chunk": "trecho 1"})
    print(f"notes  : {result['notes']}")
    print(f"history: {result['history']}")
    print("\npassos:")
    for update in app.stream({"chunk": "trecho 1"}):
        print(f"  {update}")


if __name__ == "__main__":
    main()
