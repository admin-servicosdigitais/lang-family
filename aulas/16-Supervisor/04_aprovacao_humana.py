"""Passo 4: pausar para aprovação humana com ``interrupt``."""

from typing import TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class State(TypedDict, total=False):
    amount: float
    result: str


def main() -> None:
    entries: list[str] = []
    refunds: list[float] = []

    def refund(state: State) -> State:
        entries.append("entrou no nó")
        approved = interrupt({"pergunta": "estornar?", "valor": state["amount"]})
        if approved is not True:  # só o valor True aprova
            return {"result": "recusado"}
        refunds.append(state["amount"])
        return {"result": "estornado"}

    graph = StateGraph(State)
    graph.add_node("refund", refund)
    graph.add_edge(START, "refund")
    graph.add_edge("refund", END)
    # O checkpointer guarda o estado enquanto a pessoa decide.
    app = graph.compile(checkpointer=InMemorySaver())

    for thread, answer in (("caso-1", True), ("caso-2", "sim")):
        entries.clear()
        refunds.clear()
        config: RunnableConfig = {"configurable": {"thread_id": thread}}
        paused = app.invoke({"amount": 350.0}, config)
        print(f"{thread}: pausado com {paused['__interrupt__'][0].value}")
        print(f"  estornos até aqui: {refunds}")
        final = app.invoke(Command(resume=answer), config)
        print(f"  resposta {answer!r} -> {final['result']}; o nó executou {len(entries)} vezes")
    print("\nAo retomar, o nó recomeça do início: nada com efeito pode vir antes do interrupt.")


if __name__ == "__main__":
    main()
