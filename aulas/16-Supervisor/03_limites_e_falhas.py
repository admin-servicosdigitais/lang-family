"""Passo 3: o código garante o orçamento de passos e sobrevive a falhas."""

import operator
from collections.abc import Callable
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import START, StateGraph
from langgraph.types import Command

MAX_STEPS = 3


class State(TypedDict, total=False):
    reports: Annotated[list[str], operator.add]
    steps: int
    status: str


def build(worker: Callable[[], str]) -> Any:
    def supervisor(state: State) -> Command[Literal["worker", "__end__"]]:
        reports = state.get("reports", [])
        if reports and reports[-1].startswith("ok"):
            return Command(goto="__end__", update={"status": "concluido"})
        if state.get("steps", 0) >= MAX_STEPS:
            return Command(goto="__end__", update={"status": "orcamento_esgotado"})
        return Command(goto="worker", update={"steps": state.get("steps", 0) + 1})

    def work(state: State) -> Command[Literal["supervisor"]]:
        try:
            report = f"ok: {worker()}"
        except Exception as error:
            # A falha vira informação para o supervisor decidir o que fazer.
            report = f"FALHOU: {type(error).__name__}: {error}"
        return Command(goto="supervisor", update={"reports": [report]})

    graph = StateGraph(State)
    graph.add_node("supervisor", supervisor)
    graph.add_node("worker", work)
    graph.add_edge(START, "supervisor")
    return graph.compile()


def main() -> None:
    attempts = iter([TimeoutError("sistema fora do ar")])

    def flaky() -> str:
        for error in attempts:
            raise error
        return "feito"

    def broken() -> str:
        raise TimeoutError("sistema fora do ar")

    for title, worker in (("falha uma vez", flaky), ("falha sempre", broken)):
        result = build(worker).invoke({})
        print(f"{title}: status={result['status']} passos={result['steps']}")
        for report in result["reports"]:
            print(f"  {report}")


if __name__ == "__main__":
    main()
