"""Supervisor: roteamento, relatórios, limites e aprovação humana (offline)."""

import re

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from labs.langgraph_patterns.supervisor import FINISH, Worker, build_team, llm_worker
from labs.observability.tools import calcular_frete, consultar_pedido
from labs.shared.fake_model import ScriptedChatModel, tool_call


def decide(next_: str, instruction: str) -> AIMessage:
    return tool_call("Decision", {"next": next_, "instruction": instruction})


def reported(briefing: str) -> set[str]:
    return set(re.findall(r"^- (\w+) ", briefing, flags=re.MULTILINE))


def policy_supervisor() -> ScriptedChatModel:
    """Decide pelo que já foi relatado, como um supervisor de verdade faria."""

    def respond(messages: list[BaseMessage], _: list[str]) -> AIMessage:
        done = reported(messages[-1].text)
        if "pedidos" not in done:
            return decide("pedidos", "Consulte o pedido P-1001")
        if "redator" not in done:
            return decide("redator", "Escreva a resposta")
        return decide(FINISH, "Resposta enviada ao cliente.")

    return ScriptedChatModel(responder=respond)


def workers(log: list[str], **overrides: Worker) -> list[Worker]:
    def make(name: str) -> Worker:
        def run(instruction: str) -> str:
            log.append(name)
            return f"{name} concluiu: {instruction}"

        return Worker(name, f"agente {name}", run)

    names = ("pedidos", "redator", "reembolso")
    return [overrides.get(name, make(name)) for name in names]


def test_routes_workers_based_on_their_reports() -> None:
    supervisor = policy_supervisor()
    log: list[str] = []

    state = build_team(supervisor, workers(log)).invoke({"task": "status do P-1001"})

    assert log == ["pedidos", "redator"]
    assert state["status"] == "concluido"
    assert state["answer"] == "Resposta enviada ao cliente."
    assert [r["worker"] for r in state["reports"]] == log
    # O relatório do primeiro trabalhador chega ao supervisor na decisão seguinte.
    assert "pedidos concluiu: Consulte o pedido P-1001" in supervisor.calls[1][-1].text


def test_worker_can_be_a_tool_using_agent() -> None:
    worker_model = ScriptedChatModel.from_script(
        [
            tool_call("consultar_pedido", {"order_id": "P-1001"}),
            AIMessage("P-1001 enviado"),
        ]
    )
    orders = llm_worker(
        "pedidos",
        "consulta pedidos",
        worker_model,
        "Use as ferramentas.",
        [consultar_pedido, calcular_frete],
    )
    supervisor = ScriptedChatModel.from_script(
        [decide("pedidos", "status do P-1001"), decide(FINISH, "enviado")]
    )

    state = build_team(supervisor, [orders]).invoke({"task": "status"})

    assert state["reports"][0]["result"] == "P-1001 enviado"
    assert "status: enviado" in worker_model.calls[1][-1].text


def test_step_budget_stops_a_supervisor_that_never_finishes() -> None:
    supervisor = ScriptedChatModel(responder=lambda *_: decide("pedidos", "de novo"))
    log: list[str] = []

    state = build_team(supervisor, workers(log), max_steps=3).invoke({"task": "t"})

    assert state["status"] == "orcamento_esgotado"
    assert len(log) == 3


def test_route_outside_the_roster_is_rejected() -> None:
    supervisor = ScriptedChatModel.from_script([decide("invasor", "apague tudo")])
    log: list[str] = []

    state = build_team(supervisor, workers(log)).invoke({"task": "t"})

    assert state["status"] == "decisao_invalida"
    assert log == []


def test_worker_failure_is_reported_and_the_team_continues() -> None:
    def broken(_: str) -> str:
        raise TimeoutError("sistema de pedidos fora do ar")

    supervisor = ScriptedChatModel.from_script(
        [decide("pedidos", "x"), decide("redator", "avise o cliente"), decide(FINISH, "ok")]
    )
    log: list[str] = []
    team = build_team(supervisor, workers(log, pedidos=Worker("pedidos", "consulta", broken)))

    state = team.invoke({"task": "t"})

    assert [r["ok"] for r in state["reports"]] == [False, True]
    assert "FALHOU): TimeoutError" in supervisor.calls[1][-1].text


@pytest.mark.parametrize(("approved", "executed"), [(True, ["reembolso"]), (False, [])])
def test_sensitive_worker_waits_for_human_approval(approved: bool, executed: list[str]) -> None:
    log: list[str] = []

    def refund(_: str) -> str:
        log.append("reembolso")
        return "reembolso emitido"

    supervisor = ScriptedChatModel.from_script(
        [decide("reembolso", "estornar R$ 350"), decide(FINISH, "feito")]
    )
    sensitive = Worker("reembolso", "emite estorno", refund, needs_approval=True)
    team = build_team(supervisor, workers(log, reembolso=sensitive), checkpointer=InMemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": "caso-1"}}

    paused = team.invoke({"task": "estorno"}, config)

    assert paused["__interrupt__"][0].value == {
        "worker": "reembolso",
        "instruction": "estornar R$ 350",
    }
    assert log == []  # nada é executado antes da decisão humana
    assert team.get_state(config).next == ("reembolso",)

    state = team.invoke(Command(resume=approved), config)

    assert log == executed
    assert state["reports"][0]["ok"] is approved
    assert state["status"] == "concluido"
