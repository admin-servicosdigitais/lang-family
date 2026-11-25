"""Agente LangChain/LangGraph que usa as ferramentas de um servidor MCP.

uv run python -m labs.mcp_lab.agent     # exige uma chave de LLM no ambiente
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.graph.state import CompiledStateGraph
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from labs.shared.llm import available_provider, live_chat_model

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


async def main() -> None:
    load_dotenv()
    if available_provider() is None:
        sys.exit("defina ANTHROPIC_API_KEY, OPENAI_API_KEY ou GEMINI_API_KEY")
    async with (
        stdio_client(stdio_server_params()) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        agent = await build_mcp_agent(session, live_chat_model())
        state = await agent.ainvoke({"messages": [("human", "Qual o status do pedido P-1002?")]})
        print(state["messages"][-1].text)


if __name__ == "__main__":
    asyncio.run(main())
