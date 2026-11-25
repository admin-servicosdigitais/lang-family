"""Passo 4: um agente que descobre as ferramentas no servidor, com limite de chamadas."""

import asyncio
import logging

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session
from modelo_falso import ScriptedChatModel, tool_call

server = FastMCP("aula", log_level="WARNING")


@server.tool()
def consultar_pedido(order_id: str) -> str:
    """Consulta o status de um pedido."""
    if order_id != "P-1001":
        raise ValueError("pedido não encontrado")
    return "enviado"


async def main() -> None:
    logging.getLogger("mcp").setLevel(logging.WARNING)
    limit = ModelCallLimitMiddleware(run_limit=3, exit_behavior="end")
    async with create_connected_server_and_client_session(server) as session:
        # Cada ferramenta MCP vira uma ferramenta do LangChain.
        tools = await load_mcp_tools(session)
        print(f"descobertas: {[t.name for t in tools]}")

        model = ScriptedChatModel.from_script(
            [tool_call("consultar_pedido", {"order_id": "P-1001"}), AIMessage("Enviado.")]
        )
        agent = create_agent(model, tools=tools)
        state = await agent.ainvoke({"messages": [HumanMessage("Cadê o P-1001?")]})
        print(f"resposta: {state['messages'][-1].text}")

        looping = ScriptedChatModel(
            responder=lambda *_: tool_call("consultar_pedido", {"order_id": "P-0"})
        )
        agent = create_agent(looping, tools=tools, middleware=[limit])
        state = await agent.ainvoke({"messages": [HumanMessage("oi")]})
        print(f"modelo em laço parou após {len(looping.calls)} chamadas")


if __name__ == "__main__":
    asyncio.run(main())
