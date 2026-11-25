"""Supervisor-Driven Network: um supervisor delega a agentes especializados.

    START -> supervisor -> <trabalhador> -> supervisor -> ... -> END

O supervisor decide o próximo passo com saída estruturada, a partir dos relatórios
recebidos. O código (não o modelo) garante o orçamento de passos, a validade da rota
e a aprovação humana das ações sensíveis.
"""

import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Protocol, TypedDict

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field, create_model

FINISH = "FINALIZAR"


class Report(TypedDict):
    worker: str
    instruction: str
    result: str
    ok: bool


class TeamState(TypedDict, total=False):
    task: str
    reports: Annotated[list[Report], operator.add]
    steps: int
    instruction: str
    answer: str
    status: Literal["concluido", "orcamento_esgotado", "decisao_invalida"]


@dataclass(frozen=True)
class Worker:
    name: str
    description: str
    run: Callable[[str], str]
    needs_approval: bool = False


def llm_worker(
    name: str,
    description: str,
    model: BaseChatModel,
    system_prompt: str,
    tools: Sequence[BaseTool] = (),
    *,
    needs_approval: bool = False,
) -> Worker:
    """Trabalhador que é, ele próprio, um agente LangChain com suas ferramentas."""
    agent = create_agent(model, tools=list(tools), system_prompt=system_prompt, name=name)

    def run(instruction: str) -> str:
        state = agent.invoke({"messages": [HumanMessage(instruction)]})
        return str(state["messages"][-1].text)

    return Worker(name, description, run, needs_approval)


class _Node(Protocol):
    def __call__(self, state: TeamState) -> Command[str]: ...


def _decision_schema(workers: Sequence[Worker]) -> type[BaseModel]:
    options = (*(w.name for w in workers), FINISH)
    # Enum montado em tempo de execução, conforme a equipe.
    route: Any = Literal[options]
    schema: type[BaseModel] = create_model(
        "Decision",
        __doc__="Próximo passo escolhido pelo supervisor.",
        next=(route, Field(description="Trabalhador a acionar, ou FINALIZAR")),
        instruction=(
            str,
            Field(description="Instrução ao trabalhador; ao finalizar, a resposta final"),
        ),
    )
    return schema


def _briefing(state: TeamState) -> str:
    reports = "\n".join(
        f"- {r['worker']} ({'ok' if r['ok'] else 'FALHOU'}): {r['result']}"
        for r in state.get("reports", [])
    )
    return f"Tarefa: {state['task']}\n\nRelatórios:\n{reports or '(nenhum)'}"


def build_team(
    model: BaseChatModel,
    workers: Sequence[Worker],
    *,
    max_steps: int = 6,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    schema = _decision_schema(workers)
    router = model.with_structured_output(schema)
    roster = "\n".join(f"- {w.name}: {w.description}" for w in workers)
    system = (
        "Você é o supervisor de uma equipe. Acione um trabalhador por vez e finalize "
        f"quando os relatórios bastarem para responder.\nTrabalhadores:\n{roster}"
    )
    names = [w.name for w in workers]

    def supervisor(state: TeamState) -> Command[str]:
        steps = state.get("steps", 0)
        try:
            decision: Any = router.invoke([SystemMessage(system), HumanMessage(_briefing(state))])
        except ValueError as error:  # inclui rota fora do enum (ValidationError)
            return Command(goto=END, update={"status": "decisao_invalida", "answer": str(error)})
        if decision is None:
            return Command(goto=END, update={"status": "decisao_invalida", "answer": ""})
        if decision.next == FINISH:
            return Command(goto=END, update={"status": "concluido", "answer": decision.instruction})
        if steps >= max_steps:
            return Command(goto=END, update={"status": "orcamento_esgotado", "answer": ""})
        return Command(
            goto=decision.next,
            update={"steps": steps + 1, "instruction": decision.instruction},
        )

    def node_for(worker: Worker) -> _Node:
        def node(state: TeamState) -> Command[str]:
            instruction = state["instruction"]
            if worker.needs_approval:
                # Pausa o grafo; a execução retoma com Command(resume=True | False).
                approved = interrupt({"worker": worker.name, "instruction": instruction})
                if approved is not True:
                    report = Report(
                        worker=worker.name,
                        instruction=instruction,
                        result="ação recusada pelo revisor humano",
                        ok=False,
                    )
                    return Command(goto="supervisor", update={"reports": [report]})
            try:
                result, ok = worker.run(instruction), True
            except Exception as error:
                result, ok = f"{type(error).__name__}: {error}", False
            report = Report(worker=worker.name, instruction=instruction, result=result, ok=ok)
            return Command(goto="supervisor", update={"reports": [report]})

        return node

    graph = StateGraph(TeamState)
    graph.add_node("supervisor", supervisor, destinations=(*names, END))
    for worker in workers:
        graph.add_node(worker.name, node_for(worker), destinations=("supervisor",))
    graph.add_edge(START, "supervisor")
    return graph.compile(checkpointer=checkpointer, name="equipe")
