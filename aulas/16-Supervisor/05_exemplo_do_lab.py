"""Passo 5: o supervisor do laboratório (``langgraph_patterns/supervisor.py``).

O modelo por regras e o ``main`` são só para a aula.
"""

import operator
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Protocol, TypedDict

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt
from modelo_falso import ScriptedChatModel, tool_call
from pydantic import BaseModel, Field, create_model

# --- Do laboratório: langgraph_patterns/supervisor.py -------------------------------


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
    if any(w.name == FINISH for w in workers):
        raise ValueError(f"{FINISH} é um nome reservado ao supervisor")
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
            except GraphBubbleUp:
                # Pausas e desvios do LangGraph usam exceções: não são falhas.
                raise
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


# --- Somente da aula ----------------------------------------------------------------


def policy(messages: list[BaseMessage], _: list[str]) -> AIMessage:
    """Faz o papel do modelo: decide pelo que já foi relatado."""
    done = set(re.findall(r"^- (\w+) ", messages[-1].text, flags=re.MULTILINE))
    for name, instruction in (("pedidos", "Consulte o P-1001"), ("reembolso", "Estornar R$ 350")):
        if name not in done:
            return tool_call("Decision", {"next": name, "instruction": instruction})
    return tool_call("Decision", {"next": FINISH, "instruction": "Atendimento concluído."})


def main() -> None:
    workers = [
        Worker("pedidos", "consulta pedidos", lambda _: "P-1001 chegou quebrado"),
        Worker("reembolso", "emite estornos", lambda _: "estorno emitido", needs_approval=True),
    ]
    team = build_team(ScriptedChatModel(responder=policy), workers, checkpointer=InMemorySaver())
    for thread, approved in (("caso-1", True), ("caso-2", False)):
        config: RunnableConfig = {"configurable": {"thread_id": thread}}
        paused = team.invoke({"task": "O produto chegou quebrado"}, config)
        print(f"{thread}: aguardando aprovação de {paused['__interrupt__'][0].value}")
        state = team.invoke(Command(resume=approved), config)
        for report in state["reports"]:
            print(f"  {report['worker']:<10} ok={report['ok']!s:<5} {report['result']}")
        print(f"  status={state['status']} passos={state['steps']}")


if __name__ == "__main__":
    main()
