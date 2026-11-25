"""Passo 4: memória por conversa com um *checkpointer*."""

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from modelo_falso import ScriptedChatModel


def main() -> None:
    model = ScriptedChatModel(responder=lambda messages, _: AIMessage(f"vi {len(messages)}"))
    agent = create_agent(model, checkpointer=InMemorySaver())

    for thread, text in (("ana", "meu nome é Ana"), ("ana", "qual meu nome?"), ("bia", "oi")):
        config: RunnableConfig = {"configurable": {"thread_id": thread}}
        agent.invoke({"messages": [HumanMessage(text)]}, config)
        seen = [m.text for m in model.calls[-1]]
        print(f"conversa {thread}: o modelo recebeu {seen}")

    try:
        agent.invoke({"messages": [HumanMessage("oi")]})
    except ValueError:
        print("sem thread_id: erro, porque não há onde guardar a conversa")


if __name__ == "__main__":
    main()
