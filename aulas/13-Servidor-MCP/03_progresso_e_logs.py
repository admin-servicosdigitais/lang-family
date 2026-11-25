"""Passo 3: o servidor avisa o cliente do andamento enquanto trabalha."""

import asyncio

from mcp.server.fastmcp import Context, FastMCP
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import LoggingMessageNotificationParams

server = FastMCP("aula", log_level="WARNING")


@server.tool()
async def importar_pedidos(
    quantidade: int,
    ctx: Context,  # type: ignore[type-arg]  # injetado pelo servidor, fora do schema
) -> str:
    """Importa pedidos, informando o progresso."""
    for done in range(1, quantidade + 1):
        await asyncio.sleep(0.01)
        await ctx.report_progress(done, quantidade, f"pedido {done}")
    await ctx.info(f"{quantidade} pedidos importados")
    return "concluído"


async def on_log(params: LoggingMessageNotificationParams) -> None:
    print(f"  log do servidor ({params.level}): {params.data}")


async def on_progress(progress: float, total: float | None, message: str | None) -> None:
    print(f"  progresso {progress:.0f}/{total:.0f}: {message}")


async def main() -> None:
    async with create_connected_server_and_client_session(
        server, logging_callback=on_log
    ) as session:
        tool = (await session.list_tools()).tools[0]
        print(f"argumentos visíveis ao modelo: {list(tool.inputSchema['properties'])}")
        result = await session.call_tool(
            "importar_pedidos", {"quantidade": 3}, progress_callback=on_progress
        )
        print(f"resultado: {result.structuredContent}")


if __name__ == "__main__":
    asyncio.run(main())
