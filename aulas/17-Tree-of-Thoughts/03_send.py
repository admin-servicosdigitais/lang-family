"""Passo 3: ramos paralelos no LangGraph com ``Send``."""

import operator
import threading
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


class State(TypedDict, total=False):
    numbers: list[int]
    results: Annotated[list[str], operator.add]  # junta o que cada ramo devolve


class Branch(TypedDict):
    number: int


def fan_out(state: State) -> list[Send]:
    # Um Send por item: a quantidade de ramos só é conhecida durante a execução.
    return [Send("double", Branch(number=n)) for n in state["numbers"]]


def main() -> None:
    # Cada ramo espera os outros: só termina se os três rodarem ao mesmo tempo.
    barrier = threading.Barrier(3, timeout=5)

    # O parâmetro precisa se chamar "state" para o tipo bater com o LangGraph.
    def double(state: Branch) -> State:
        barrier.wait()
        return {"results": [f"{state['number']} -> {state['number'] * 2}"]}

    def summarize(state: State) -> State:
        print(f"o nó seguinte roda uma vez, com tudo: {state['results']}")
        return {}

    graph = StateGraph(State)
    graph.add_node("double", double, input_schema=Branch)
    graph.add_node("summarize", summarize)
    graph.add_conditional_edges(START, fan_out, ["double"])
    graph.add_edge("double", "summarize")
    graph.add_edge("summarize", END)
    graph.compile().invoke({"numbers": [2, 7, 9]})


if __name__ == "__main__":
    main()
