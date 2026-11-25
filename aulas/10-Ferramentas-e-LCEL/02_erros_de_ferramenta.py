"""Passo 2: erro previsto vira mensagem para o modelo, não exceção."""

from langchain_core.messages import ToolCall
from langchain_core.tools import ToolException, tool

ORDERS = {"P-1001": "enviado", "P-1002": "aguardando pagamento"}


@tool
def consultar_pedido(order_id: str) -> str:
    """Consulta o status de um pedido, pelo número (ex.: P-1001)."""
    status = ORDERS.get(order_id.strip().upper())
    if status is None:
        raise ToolException(f"pedido {order_id} não encontrado")
    return status


def call(order_id: str) -> ToolCall:
    """O formato em que o pedido do modelo chega (aula 03)."""
    return ToolCall(name="consultar_pedido", args={"order_id": order_id}, id="1", type="tool_call")


def main() -> None:
    try:
        consultar_pedido.invoke(call("P-0"))
    except ToolException as error:
        print(f"sem tratamento: exceção -> {error}")

    consultar_pedido.handle_tool_error = True
    for order_id in ("P-1001", "P-0"):
        message = consultar_pedido.invoke(call(order_id))
        print(f"com tratamento: {type(message).__name__} status={message.status} {message.content}")


if __name__ == "__main__":
    main()
