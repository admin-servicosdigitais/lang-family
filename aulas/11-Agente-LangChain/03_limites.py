"""Passo 3: limitar chamadas para um agente em laço não gastar sem parar."""

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from modelo_falso import ScriptedChatModel, tool_call


@tool
def consultar_pedido(order_id: str) -> str:
    """Consulta o status de um pedido."""
    return "enviado"


def run(title: str, middleware: list[Any]) -> None:
    # Um modelo defeituoso: pede a mesma ferramenta para sempre.
    model = ScriptedChatModel(
        responder=lambda _, __: tool_call("consultar_pedido", {"order_id": "P-1001"})
    )
    agent = create_agent(model, tools=[consultar_pedido], middleware=middleware)
    state = agent.invoke({"messages": [HumanMessage("oi")]})
    results = [m.status for m in state["messages"] if isinstance(m, ToolMessage)]
    print(f"{title}\n  chamadas ao modelo={len(model.calls)} ferramentas={results}")
    print(f"  última mensagem: {state['messages'][-1].text}")


def main() -> None:
    run("só limite de modelo (3)", [ModelCallLimitMiddleware(run_limit=3, exit_behavior="end")])
    run(
        "modelo (4) e ferramentas (2)",
        [
            ModelCallLimitMiddleware(run_limit=4, exit_behavior="end"),
            ToolCallLimitMiddleware(run_limit=2),
        ],
    )


if __name__ == "__main__":
    main()
