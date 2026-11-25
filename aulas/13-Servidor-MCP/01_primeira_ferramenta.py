"""Passo 1: um servidor MCP com uma ferramenta, consultado por um cliente em memória."""

import asyncio
import json

from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import TextContent

server = FastMCP("aula", log_level="WARNING")


@server.tool()
def consultar_pedido(order_id: str) -> str:
    """Consulta o status de um pedido (ex.: P-1001)."""
    return {"P-1001": "enviado"}.get(order_id, "não encontrado")


async def main() -> None:
    # Cliente e servidor conversam pelo protocolo, sem processo nem rede.
    async with create_connected_server_and_client_session(server) as session:
        for tool in (await session.list_tools()).tools:
            print(f"ferramenta: {tool.name} - {tool.description}")
            print(f"schema    : {json.dumps(tool.inputSchema, ensure_ascii=False)}")
        result = await session.call_tool("consultar_pedido", {"order_id": "P-1001"})
        texts = [c.text for c in result.content if isinstance(c, TextContent)]
        print(f"resultado : {texts} erro={result.isError}")


if __name__ == "__main__":
    asyncio.run(main())
