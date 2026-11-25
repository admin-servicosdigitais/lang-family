"""Servidor MCP da Loja Aurora: ferramentas, recursos e prompt.

uv run python -m labs.mcp_lab.server                      # stdio
uv run python -m labs.mcp_lab.server --transport streamable-http --port 8765
"""

import argparse
from typing import Annotated, Literal

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field, StringConstraints

from labs.rag.chunking import DATA_DIR

ORDERS: dict[str, tuple[str, float]] = {
    "P-1001": ("enviado", 350.0),
    "P-1002": ("aguardando pagamento", 120.0),
}


class Order(BaseModel):
    order_id: str
    status: str
    total: float


class Ticket(BaseModel):
    ticket_id: str
    order_id: str
    reason: str
    priority: Literal["baixa", "normal", "alta"]


def _find_order(order_id: str) -> Order:
    order_id = order_id.strip().upper()
    if order_id not in ORDERS:
        raise ValueError(f"pedido {order_id} não encontrado")
    status, total = ORDERS[order_id]
    return Order(order_id=order_id, status=status, total=total)


def _policies() -> dict[str, str]:
    """Lista fechada de documentos: o nome pedido nunca vira caminho de arquivo."""
    return {path.stem: path.read_text(encoding="utf-8") for path in DATA_DIR.glob("*.md")}


def create_server(host: str = "127.0.0.1", port: int = 8765) -> FastMCP:
    """Cada instância tem sua própria fila de chamados (isola os testes)."""
    server = FastMCP(
        "loja-aurora",
        instructions="Consulta pedidos e políticas e abre chamados de atendimento.",
        host=host,
        port=port,
    )
    tickets: list[Ticket] = []

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def consultar_pedido(order_id: str) -> Order:
        """Consulta status e valor de um pedido (ex.: P-1001)."""
        return _find_order(order_id)

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False))
    async def abrir_chamado(
        order_id: str,
        ctx: Context,  # type: ignore[type-arg]
        # Espaços são removidos antes de medir o tamanho mínimo.
        reason: Annotated[
            str,
            StringConstraints(strip_whitespace=True, min_length=10),
            Field(description="Descrição do problema"),
        ],
        priority: Literal["baixa", "normal", "alta"] = "normal",
    ) -> Ticket:
        """Abre um chamado de atendimento para um pedido existente."""
        await ctx.report_progress(0, 2, "validando pedido")
        order = _find_order(order_id)
        ticket = Ticket(
            ticket_id=f"C-{len(tickets) + 1:04d}",
            order_id=order.order_id,
            reason=reason,
            priority=priority,
        )
        tickets.append(ticket)
        await ctx.info(f"chamado {ticket.ticket_id} aberto")
        await ctx.report_progress(2, 2, "concluído")
        return ticket

    @server.resource("politicas://indice", mime_type="text/plain")
    def indice_de_politicas() -> str:
        """Nomes das políticas disponíveis, um por linha."""
        return "\n".join(sorted(_policies()))

    @server.resource("politicas://{name}", mime_type="text/markdown")
    def politica(name: str) -> str:
        """Texto completo de uma política da loja."""
        policies = _policies()
        if name not in policies:
            raise ValueError(f"política desconhecida: {name}")
        return policies[name]

    @server.prompt()
    def responder_cliente(order_id: str) -> str:
        """Roteiro para responder um cliente sobre um pedido."""
        # Só um pedido existente entra no texto: o argumento não vira instrução livre.
        order_id = _find_order(order_id).order_id
        return (
            f"Consulte o pedido {order_id} com a ferramenta consultar_pedido e escreva ao "
            "cliente uma resposta cordial, citando apenas dados retornados."
        )

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    create_server(port=args.port).run(args.transport)


if __name__ == "__main__":
    main()
