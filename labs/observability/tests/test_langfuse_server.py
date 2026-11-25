"""Ida e volta em um servidor Langfuse real (self-hosted): grava e lê pela API.

Suba antes: ``docker compose -f infra/langfuse.compose.yml up -d --wait``
"""

import time
import uuid
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage
from langfuse import Langfuse, propagate_attributes

from labs.observability.agent import build_agent
from labs.observability.tracing import RedactingLangfuseHandler, langfuse_client
from labs.shared.fake_model import ScriptedChatModel, tool_call

pytestmark = pytest.mark.docker

URL = "http://localhost:13000"
# Credenciais fixas de desenvolvimento, definidas no arquivo compose.
PUBLIC_KEY = "pk-lf-local-dev"
SECRET_KEY = "sk-lf-local-dev"
CPF = "123.456.789-09"


@pytest.fixture(scope="module")
def client() -> Iterator[Langfuse]:
    try:
        httpx.get(f"{URL}/api/public/health", timeout=2).raise_for_status()
    except httpx.HTTPError:
        pytest.skip(f"Langfuse indisponível em {URL} (suba infra/langfuse.compose.yml)")
    langfuse = langfuse_client(public_key=PUBLIC_KEY, secret_key=SECRET_KEY, base_url=URL)
    yield langfuse
    langfuse.shutdown()


def wait_for[T](fetch: Callable[[], T], ready: Callable[[T], bool], timeout: float = 60) -> T:
    """A ingestão é assíncrona: consulta até o dado aparecer."""
    deadline = time.monotonic() + timeout
    while True:
        result = fetch()
        if ready(result) or time.monotonic() > deadline:
            assert ready(result), f"dados não apareceram em {timeout}s"
            return result
        time.sleep(1)


def test_trace_and_score_round_trip(client: Langfuse) -> None:
    trace_id = Langfuse.create_trace_id()
    session_id = f"sessao-{uuid.uuid4().hex[:8]}"
    model = ScriptedChatModel.from_script(
        [tool_call("consultar_pedido", {"order_id": "P-1001"}), AIMessage("Enviado.")]
    )
    handler = RedactingLangfuseHandler(public_key=PUBLIC_KEY, trace_context={"trace_id": trace_id})

    with propagate_attributes(session_id=session_id, user_id="usuario-teste"):
        build_agent(model).invoke(
            {"messages": [("human", f"CPF {CPF}: status do P-1001?")]}, {"callbacks": [handler]}
        )
    client.create_score(trace_id=trace_id, name="resolvido", value=1.0)
    client.flush()

    def observations() -> list[Any]:
        page = client.api.observations.get_many(
            trace_id=trace_id, fields="core,basic,io,model,usage,trace_context", limit=100
        )
        return list(page.data)

    found = wait_for(
        observations,
        lambda items: (
            {"GENERATION", "TOOL"} <= {o.type for o in items}
            and sum(o.type == "GENERATION" for o in items) == 2
        ),
    )
    tool = next(o for o in found if o.type == "TOOL")
    generation = next(o for o in found if o.type == "GENERATION")
    assert tool.name == "consultar_pedido"
    assert generation.usage_details is not None
    assert generation.usage_details["total"] > 0
    assert {o.session_id for o in found} == {session_id}
    stored = str([o.model_dump() for o in found])
    assert "[CPF]" in stored
    assert CPF not in stored

    scores = wait_for(
        lambda: list(client.api.scores.get_many(trace_id=trace_id).data), lambda s: bool(s)
    )
    assert [(s.name, s.value) for s in scores] == [("resolvido", 1.0)]
