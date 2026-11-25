"""Ferramentas LangChain do agente de atendimento (dados fictícios em memória)."""

from langchain_core.tools import ToolException, tool

ORDERS = {
    "P-1001": {"status": "enviado", "total": 350.0, "regiao": "sudeste"},
    "P-1002": {"status": "aguardando pagamento", "total": 120.0, "regiao": "norte"},
}


@tool
def consultar_pedido(order_id: str) -> str:
    """Consulta status, valor e região de um pedido, pelo número (ex.: P-1001)."""
    order = ORDERS.get(order_id.strip().upper())
    if order is None:
        raise ToolException(f"pedido {order_id} não encontrado")
    return f"status: {order['status']}; total: R$ {order['total']:.2f}; região: {order['regiao']}"


@tool
def calcular_frete(total: float, regiao: str) -> str:
    """Calcula o frete a partir do valor do pedido e da região do país."""
    if total > 299 and regiao.strip().lower() in {"sul", "sudeste"}:
        return "frete grátis"
    return "frete: R$ 29,90"


# Erros previstos voltam ao modelo como mensagem, sem derrubar o agente.
consultar_pedido.handle_tool_error = True

TOOLS = [consultar_pedido, calcular_frete]
