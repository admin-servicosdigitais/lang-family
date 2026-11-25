"""Passo 1: um supervisor que decide o próximo nó com ``Command``."""

import operator
from typing import Annotated, Literal, TypedDict

from langgraph.graph import START, StateGraph
from langgraph.types import Command


class State(TypedDict, total=False):
    reports: Annotated[list[str], operator.add]


def supervisor(state: State) -> Command[Literal["pedidos", "redator", "__end__"]]:
    # Aqui a decisão é uma regra; no passo 2 ela passa a ser do modelo.
    done = len(state.get("reports", []))
    if done == 0:
        return Command(goto="pedidos")
    if done == 1:
        return Command(goto="redator")
    return Command(goto="__end__")


def orders(state: State) -> Command[Literal["supervisor"]]:
    return Command(goto="supervisor", update={"reports": ["pedido enviado"]})


def writer(state: State) -> Command[Literal["supervisor"]]:
    return Command(goto="supervisor", update={"reports": ["resposta escrita"]})


def main() -> None:
    graph = StateGraph(State)
    graph.add_node("supervisor", supervisor)
    graph.add_node("pedidos", orders)
    graph.add_node("redator", writer)
    # Só a entrada é uma aresta fixa; o resto do caminho vem dos Commands.
    graph.add_edge(START, "supervisor")
    app = graph.compile()
    for step in app.stream({}):
        print(step)


if __name__ == "__main__":
    main()
