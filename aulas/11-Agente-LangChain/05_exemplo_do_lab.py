"""Passo 5: o agente do laboratório (``labs/observability/agent.py``).

As ferramentas (aula 10) vêm junto. O ``main`` é só para a aula.
"""

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import ToolException, tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from modelo_falso import ScriptedChatModel, tool_call

# --- Da aula 10: tools.py -----------------------------------------------------------


ORDERS = {
    "P-1001": {"status": "enviado", "total": 350.0, "regiao": "sudeste"},
    "P-1002": {"status": "aguardando pagamento", "total": 120.0, "regiao": "norte"},
}


@tool
def consultar_pedido(order_id: str) -> str:
    """Consulta status, valor e região de um pedido, pelo número (ex.: P-1001)."""
    order = ORDERS.get(order_id.strip().upper())
    if order is None:
        raise ToolException(f"pedido {order_id} não encontrado")
    return f"status: {order['status']}; total: R$ {order['total']:.2f}; região: {order['regiao']}"


@tool
def calcular_frete(total: float, regiao: str) -> str:
    """Calcula o frete a partir do valor do pedido e da região do país."""
    if total > 299 and regiao.strip().lower() in {"sul", "sudeste"}:
        return "frete grátis"
    return "frete: R$ 29,90"


consultar_pedido.handle_tool_error = True

TOOLS = [consultar_pedido, calcular_frete]


# --- Do laboratório: observability/agent.py -----------------------------------------


SYSTEM_PROMPT = (
    "Você é o atendimento da Loja Aurora. Use as ferramentas para obter dados de pedidos; "
    "nunca invente status ou valores. Responda em português, de forma curta."
)


def build_agent(
    model: BaseChatModel,
    *,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    max_model_calls: int = 6,
    max_tool_calls: int = 5,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    # Proteções contra laços infinitos (e contas altas).
    limits: list[Any] = [
        ModelCallLimitMiddleware(run_limit=max_model_calls, exit_behavior="end"),
        ToolCallLimitMiddleware(run_limit=max_tool_calls),
    ]
    return create_agent(
        model,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        middleware=limits,
        checkpointer=checkpointer,
        name="atendimento",
    )


# --- Somente da aula ----------------------------------------------------------------


def main() -> None:
    model = ScriptedChatModel.from_script(
        [
            tool_call("consultar_pedido", {"order_id": "P-1001"}),
            tool_call("calcular_frete", {"total": 350.0, "regiao": "sudeste"}),
            AIMessage("O pedido foi enviado, com frete grátis."),
            tool_call("consultar_pedido", {"order_id": "P-0"}),
            AIMessage("Não encontrei o pedido P-0."),
        ]
    )
    agent = build_agent(model, checkpointer=InMemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": "cliente-1"}}
    for question in ("Status e frete do P-1001?", "E o P-0?"):
        state = agent.invoke({"messages": [("human", question)]}, config)
        print(f"pergunta: {question}")
        print(f"resposta: {state['messages'][-1].text}")
    print("histórico da conversa:")
    for message in state["messages"]:
        calls = [c["name"] for c in getattr(message, "tool_calls", [])]
        print(f"  {message.type:<5} {message.text or calls}")


if __name__ == "__main__":
    main()
