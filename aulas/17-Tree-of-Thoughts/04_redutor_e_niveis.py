"""Passo 4: um redutor que acumula os ramos e esvazia a cada nível da árvore."""

from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

BREADTH = 2
TARGET = 20


def collect(current: list[int], update: list[int] | None) -> list[int]:
    """Acumula o que os ramos devolvem; None limpa para o próximo nível."""
    return [] if update is None else current + update


class State(TypedDict, total=False):
    frontier: list[int]
    candidates: Annotated[list[int], collect]
    depth: int


class Branch(TypedDict):
    value: int


def fan_out(state: State) -> list[Send] | str:
    if TARGET in state.get("frontier", []) or state.get("depth", 0) >= 6:
        return END
    return [Send("expand", Branch(value=v)) for v in state["frontier"]]


def expand(state: Branch) -> State:
    return {"candidates": [state["value"] + 3, state["value"] * 2]}


def prune(state: State) -> State:
    # Mantém os BREADTH estados mais próximos do alvo.
    best = sorted(set(state["candidates"]), key=lambda v: abs(TARGET - v))[:BREADTH]
    print(f"nível {state.get('depth', 0) + 1}: candidatos={state['candidates']} mantidos={best}")
    return {"frontier": best, "candidates": None, "depth": state.get("depth", 0) + 1}  # type: ignore[typeddict-item]


def main() -> None:
    graph = StateGraph(State)
    graph.add_node("expand", expand, input_schema=Branch)
    graph.add_node("prune", prune)
    graph.add_conditional_edges(START, fan_out, ["expand", END])
    graph.add_edge("expand", "prune")
    graph.add_conditional_edges("prune", fan_out, ["expand", END])
    result = graph.compile().invoke({"frontier": [1]})
    print(f"chegou a {TARGET}? {TARGET in result['frontier']} em {result['depth']} níveis")


if __name__ == "__main__":
    main()
