"""Um LLM real usando o servidor MCP: ``uv run pytest --live labs/mcp_lab``."""

import pytest
from langchain_core.messages import ToolMessage
from mcp import ClientSession
from mcp.client.stdio import stdio_client

from labs.mcp_lab.agent import build_mcp_agent, stdio_server_params
from labs.shared.llm import live_chat_model

pytestmark = pytest.mark.live


async def test_real_model_calls_the_mcp_tool() -> None:
    model = live_chat_model()

    async with (
        stdio_client(stdio_server_params()) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        agent = await build_mcp_agent(session, model)
        state = await agent.ainvoke({"messages": [("human", "Qual o status do pedido P-1002?")]})

    assert any(isinstance(m, ToolMessage) for m in state["messages"])
    assert "pagamento" in state["messages"][-1].text.lower()
