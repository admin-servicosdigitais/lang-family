"""Comportamento do agente e da cadeia LCEL (offline, modelo roteirizado)."""

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from labs.observability.agent import SYSTEM_PROMPT, build_agent
from labs.observability.tracing import MetricsCallback
from labs.observability.triage import build_triage_chain
from labs.shared.fake_model import ScriptedChatModel, tool_call


def test_agent_chains_tools_and_feeds_results_back() -> None:
    model = ScriptedChatModel.from_script(
        [
            tool_call("consultar_pedido", {"order_id": "P-1001"}),
            tool_call("calcular_frete", {"total": 350.0, "regiao": "sudeste"}),
            AIMessage("Enviado, frete grátis."),
        ]
    )
    metrics = MetricsCallback()

    state = build_agent(model).invoke(
        {"messages": [("human", "status e frete do P-1001")]}, {"callbacks": [metrics]}
    )

    outputs = [m.text for m in state["messages"] if isinstance(m, ToolMessage)]
    assert outputs[0].startswith("status: enviado")
    assert outputs[1] == "frete grátis"
    assert state["messages"][-1].text == "Enviado, frete grátis."
    assert model.calls[0][0].text == SYSTEM_PROMPT
    assert metrics.metrics.llm_calls == 3
    assert metrics.metrics.tool_calls == ["consultar_pedido", "calcular_frete"]
    assert metrics.metrics.input_tokens > 0


def test_tool_error_returns_to_model_instead_of_crashing() -> None:
    model = ScriptedChatModel.from_script(
        [tool_call("consultar_pedido", {"order_id": "P-0"}), AIMessage("Não encontrei.")]
    )
    metrics = MetricsCallback()

    state = build_agent(model).invoke(
        {"messages": [("human", "pedido P-0")]}, {"callbacks": [metrics]}
    )

    failed = next(m for m in state["messages"] if isinstance(m, ToolMessage))
    assert failed.status == "error"
    assert "não encontrado" in failed.text
    assert metrics.metrics.errors


def test_model_call_limit_stops_a_looping_agent() -> None:
    model = ScriptedChatModel(
        responder=lambda _, __: tool_call("consultar_pedido", {"order_id": "P-1001"})
    )

    state = build_agent(model, max_model_calls=3, max_tool_calls=10).invoke(
        {"messages": [("human", "loop")]}
    )

    assert len(model.calls) == 3
    assert "limit" in state["messages"][-1].text.lower()


def test_tool_call_limit_blocks_extra_calls() -> None:
    model = ScriptedChatModel.from_script(
        [
            tool_call("consultar_pedido", {"order_id": "P-1001"}),
            tool_call("consultar_pedido", {"order_id": "P-1002"}),
            AIMessage("fim"),
        ]
    )

    state = build_agent(model, max_tool_calls=1).invoke({"messages": [("human", "dois")]})

    results = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert [m.status for m in results] == ["success", "error"]


def test_memory_is_isolated_by_thread() -> None:
    model = ScriptedChatModel(responder=lambda _, __: AIMessage("ok"))
    agent = build_agent(model, checkpointer=InMemorySaver())

    agent.invoke({"messages": [("human", "sou a Ana")]}, {"configurable": {"thread_id": "1"}})
    agent.invoke({"messages": [("human", "e agora?")]}, {"configurable": {"thread_id": "1"}})
    agent.invoke({"messages": [("human", "olá")]}, {"configurable": {"thread_id": "2"}})

    assert "sou a Ana" in [m.text for m in model.calls[1]]
    assert "sou a Ana" not in [m.text for m in model.calls[2]]


@pytest.mark.parametrize(
    ("category", "urgent", "queue"),
    [
        ("pedido", False, "agente-pedidos"),
        ("financeiro", False, "financeiro"),
        ("outro", False, "faq"),
        ("pedido", True, "fila-humana"),
    ],
)
def test_triage_chain_routes_by_structured_output(category: str, urgent: bool, queue: str) -> None:
    model = ScriptedChatModel.from_script(
        [tool_call("Triage", {"category": category, "urgent": urgent})]
    )

    assert build_triage_chain(model).invoke({"message": "mensagem"}) == queue
    assert model.calls[0][-1].text == "mensagem"
