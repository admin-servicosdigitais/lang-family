"""Passo 2: o ``create_agent`` monta o laço como um grafo."""

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
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
    agent = create_agent(model, tools=[consultar_pedido], system_prompt="Responda curto.")
    print(f"nós do grafo: {list(agent.get_graph().nodes)}")

    state = agent.invoke({"messages": [HumanMessage("Cadê o P-1001?")]})
    print("estado final:")
    for message in state["messages"]:
        calls = [c["name"] for c in getattr(message, "tool_calls", [])]
        print(f"  {message.type:<5} {message.text or calls}")
    print(f"o modelo recebeu primeiro: {model.calls[0][0].type} = {model.calls[0][0].text}")


if __name__ == "__main__":
    main()
