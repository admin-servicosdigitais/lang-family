"""Passo 1: um agente é um laço: modelo, ferramentas, modelo, até a resposta."""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from modelo_falso import ScriptedChatModel, tool_call


@tool
def consultar_pedido(order_id: str) -> str:
    """Consulta o status de um pedido."""
    return {"P-1001": "enviado"}.get(order_id, "não encontrado")


def main() -> None:
    model = ScriptedChatModel.from_script(
        [
            tool_call("consultar_pedido", {"order_id": "P-1001"}),
            AIMessage("Seu pedido foi enviado."),
        ]
    )
    with_tools = model.bind_tools([consultar_pedido])
    messages: list[BaseMessage] = [HumanMessage("Cadê o P-1001?")]
    for _ in range(5):  # sem um limite, o laço poderia não terminar
        reply = with_tools.invoke(messages)
        messages.append(reply)
        if not reply.tool_calls:
            break
        for call in reply.tool_calls:
            messages.append(consultar_pedido.invoke(call))  # devolve um ToolMessage

    for message in messages:
        calls = [c["name"] for c in getattr(message, "tool_calls", [])]
        print(f"{message.type:<5} {message.text or calls}")


if __name__ == "__main__":
    main()
