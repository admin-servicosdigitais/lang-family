"""Passo 1: o ciclo de uma sessão MCP: inicializar, descobrir e chamar."""

import asyncio

from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session

server = FastMCP("aula", log_level="WARNING")


@server.tool()
def somar(a: int, b: int) -> int:
    """Soma dois números."""
    return a + b


async def main() -> None:
    # Este atalho em memória já faz o "initialize"; nos próximos passos ele é explícito.
    async with create_connected_server_and_client_session(server) as session:
        print("1. sessão inicializada")
        tools = (await session.list_tools()).tools
        print(f"2. descoberta: {[t.name for t in tools]}")
        result = await session.call_tool("somar", {"a": 2, "b": 3})
        print(f"3. chamada: {result.structuredContent}")
    print("4. sessão encerrada: o cliente não pode mais chamar nada")


if __name__ == "__main__":
    asyncio.run(main())
