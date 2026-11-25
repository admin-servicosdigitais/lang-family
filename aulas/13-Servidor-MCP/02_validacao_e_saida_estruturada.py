"""Passo 2: validar argumentos, devolver dados estruturados e anotar a ferramenta."""

import asyncio
from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import TextContent, ToolAnnotations
from pydantic import BaseModel, Field, StringConstraints

server = FastMCP("aula", log_level="WARNING")


class Ticket(BaseModel):
    order_id: str
    reason: str


@server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False))
def abrir_chamado(
    order_id: str,
    reason: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=10),
        Field(description="Descrição do problema"),
    ],
) -> Ticket:
    """Abre um chamado para um pedido existente."""
    if order_id != "P-1001":
        raise ValueError(f"pedido {order_id} não encontrado")
    return Ticket(order_id=order_id, reason=reason)


async def main() -> None:
    cases: list[tuple[str, dict[str, Any]]] = [
        ("válido", {"order_id": "P-1001", "reason": "  produto quebrado  "}),
        ("só espaços", {"order_id": "P-1001", "reason": " " * 20}),
        ("campo ausente", {"order_id": "P-1001"}),
        ("pedido errado", {"order_id": "P-9", "reason": "produto quebrado"}),
    ]
    async with create_connected_server_and_client_session(server) as session:
        tool = (await session.list_tools()).tools[0]
        print(f"anotações: {tool.annotations}")
        print(f"schema de saída: {sorted((tool.outputSchema or {})['properties'])}\n")
        for label, arguments in cases:
            result = await session.call_tool("abrir_chamado", arguments)
            if result.isError:
                first = result.content[0]
                assert isinstance(first, TextContent)
                lines = first.text.splitlines()
                print(f"{label:<14} erro: {lines[-2] if len(lines) > 2 else lines[0]}")
            else:
                print(f"{label:<14} ok  : {result.structuredContent}")
        print("\nO servidor continua no ar: falha de ferramenta é um resultado.")


if __name__ == "__main__":
    asyncio.run(main())
