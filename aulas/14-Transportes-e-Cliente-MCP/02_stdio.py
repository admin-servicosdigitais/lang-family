"""Passo 2: transporte stdio. O cliente inicia o servidor como um subprocesso.

Este arquivo faz os dois papéis: com ``--server`` ele é o servidor.
"""

import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.fastmcp import FastMCP

server = FastMCP("aula-stdio", log_level="WARNING")


@server.tool()
def quem_sou() -> int:
    """Número do processo do servidor."""
    return os.getpid()


async def client() -> None:
    parameters = StdioServerParameters(command=sys.executable, args=[__file__, "--server"])
    async with (
        stdio_client(parameters) as (read, write),
        ClientSession(read, write) as session,
    ):
        info = await session.initialize()
        result = await session.call_tool("quem_sou", {})
        assert result.structuredContent is not None
        print(f"servidor: {info.serverInfo.name}, protocolo {info.protocolVersion}")
        print(f"processos diferentes? {result.structuredContent['result'] != os.getpid()}")
    # As mensagens trafegam em JSON pela entrada e saída padrão do subprocesso.
    print("um print() no servidor corromperia o protocolo: logs vão para stderr")


if __name__ == "__main__":
    if "--server" in sys.argv:
        server.run("stdio")
    else:
        asyncio.run(client())
