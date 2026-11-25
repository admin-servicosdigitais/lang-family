"""Agente com LLM real e LangSmith real: ``uv run pytest --live labs/observability``."""

import time
import uuid

import pytest
from langchain_core.runnables import RunnableConfig

from labs.observability.agent import build_agent
from labs.observability.tracing import langsmith_client, langsmith_tracer
from labs.observability.triage import build_triage_chain
from labs.shared.llm import live_chat_model
from labs.shared.settings import require_env

pytestmark = pytest.mark.live


def test_real_model_uses_the_tools() -> None:
    state = build_agent(live_chat_model()).invoke(
        {"messages": [("human", "Qual o status do pedido P-1002?")]}
    )

    assert "pagamento" in state["messages"][-1].text.lower()


def test_real_triage() -> None:
    chain = build_triage_chain(live_chat_model())

    queue = chain.invoke({"message": "Queria a segunda via do boleto, sem pressa."})

    assert queue == "financeiro"


def test_langsmith_stores_the_trace() -> None:
    require_env("LANGSMITH_API_KEY")
    model = live_chat_model()
    client = langsmith_client()
    tracer = langsmith_tracer(client, project="lang-family-tests")
    run_id = uuid.uuid4()
    config: RunnableConfig = {"callbacks": [tracer], "run_id": run_id}

    build_agent(model).invoke({"messages": [("human", "Status do P-1001?")]}, config)
    tracer.wait_for_futures()
    client.flush()

    deadline = time.monotonic() + 60
    while True:
        try:
            run = client.read_run(run_id)
            break
        except Exception:
            if time.monotonic() > deadline:
                raise
            time.sleep(2)
    assert run.name == "atendimento"
