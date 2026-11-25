"""Executa o agente e envia o trace aos backends configurados no ambiente.

Sem chave de LLM, usa um modelo roteirizado: dá para ver o trace no Langfuse
local sem criar conta em lugar nenhum. Veja o README deste laboratório.

    uv run python -m labs.observability.demo
"""

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tracers.langchain import wait_for_all_tracers
from langfuse import get_client, propagate_attributes

from labs.observability.agent import build_agent
from labs.observability.tracing import MetricsCallback, callbacks_from_env
from labs.shared.fake_model import ScriptedChatModel, tool_call
from labs.shared.llm import available_provider, live_chat_model

QUESTION = "Qual o status do pedido P-1001 e quanto fica o frete?"


def scripted_model() -> BaseChatModel:
    return ScriptedChatModel.from_script(
        [
            tool_call("consultar_pedido", {"order_id": "P-1001"}),
            tool_call("calcular_frete", {"total": 350.0, "regiao": "sudeste"}),
            AIMessage("O pedido P-1001 foi enviado e tem frete grátis."),
        ]
    )


def main() -> None:
    load_dotenv()
    model = live_chat_model() if available_provider() else scripted_model()
    metrics = MetricsCallback()
    tracers = callbacks_from_env()
    config: RunnableConfig = {"callbacks": [metrics, *tracers], "run_name": "demo-atendimento"}

    with propagate_attributes(session_id="demo", user_id="usuario-demo", tags=["demo"]):
        state = build_agent(model).invoke({"messages": [("human", QUESTION)]}, config)

    print(state["messages"][-1].text)
    print(metrics.metrics)
    names = [type(t).__name__ for t in tracers] or ["nenhum (configure o .env)"]
    print("traces enviados para:", ", ".join(names))
    wait_for_all_tracers()
    if any(name == "RedactingLangfuseHandler" for name in names):
        get_client().flush()


if __name__ == "__main__":
    main()
