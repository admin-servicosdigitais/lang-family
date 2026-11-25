"""Passo 5: o agente MCP do laboratório (``labs/mcp_lab/agent.py``).

Inicia o servidor do laboratório por stdio. O modelo falso e o ``main`` são só para a
aula (no laboratório o ``main`` usa um modelo real e exige chave).
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.graph.state import CompiledStateGraph
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from modelo_falso import ScriptedChatModel, tool_call

# --- Do laboratório: mcp_lab/agent.py -----------------------------------------------


ROOT = Path(__file__).parents[2]


def stdio_server_params() -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable, args=["-m", "labs.mcp_lab.server"], cwd=ROOT
    )


async def build_mcp_agent(
    session: ClientSession, model: BaseChatModel, *, max_model_calls: int = 6
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Descobre as ferramentas no servidor; o agente não as conhece de antemão."""
    tools = await load_mcp_tools(session)
    return create_agent(
        model,
        tools=tools,
        system_prompt="Use as ferramentas para responder sobre pedidos.",
        # Sem limite, um modelo em laço faria milhares de chamadas pagas.
        middleware=[ModelCallLimitMiddleware(run_limit=max_model_calls, exit_behavior="end")],
    )


# --- Somente da aula ----------------------------------------------------------------


async def main() -> None:
    model = ScriptedChatModel.from_script(
        [
            tool_call("consultar_pedido", {"order_id": "P-1002"}),
            AIMessage("O pedido P-1002 está aguardando pagamento."),
        ]
    )
    async with (
        stdio_client(stdio_server_params()) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        agent = await build_mcp_agent(session, model)
        state = await agent.ainvoke({"messages": [HumanMessage("Status do P-1002?")]})
        for message in state["messages"]:
            calls = [c["name"] for c in getattr(message, "tool_calls", [])]
            print(f"{message.type:<5} {message.text or calls}")


if __name__ == "__main__":
    asyncio.run(main())
