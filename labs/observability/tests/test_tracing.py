"""Os traces que saem do processo: estrutura, atributos e ausência de PII (offline)."""

import json
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from langchain_core.language_models import FakeListLLM
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langfuse import propagate_attributes
from langgraph.types import Command
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from labs.observability.agent import build_agent
from labs.observability.tracing import (
    MetricsCallback,
    langfuse_client,
    langfuse_handler,
    langsmith_client,
    langsmith_tracer,
    redact,
)
from labs.shared.fake_http import fake_server
from labs.shared.fake_model import ScriptedChatModel, tool_call

CPF = "123.456.789-09"
EMAIL = "cliente@exemplo.com"
QUESTION = f"Sou o titular do CPF {CPF} ({EMAIL}). Status do P-1001?"


def run_agent(config: RunnableConfig) -> None:
    model = ScriptedChatModel.from_script(
        [
            tool_call("consultar_pedido", {"order_id": "P-1001"}),
            AIMessage(f"Pedido enviado. Aviso para {EMAIL}."),
        ]
    )
    build_agent(model).invoke({"messages": [("human", QUESTION)]}, config)


def test_redact_walks_nested_structures() -> None:
    data = {"a": [f"cpf {CPF}", {"b": EMAIL}], "c": 10, "d": "12345678909"}

    assert redact(data) == {"a": ["cpf [CPF]", {"b": "[EMAIL]"}], "c": 10, "d": "[CPF]"}


def test_redact_covers_dictionary_keys() -> None:
    assert redact({EMAIL: {CPF: 1}, 7: "ok"}) == {"[EMAIL]": {"[CPF]": 1}, 7: "ok"}


@pytest.mark.parametrize("text", ["cpf_12345678909", "12345678909_nota", f"doc_{CPF}.pdf"])
def test_redact_finds_cpf_next_to_underscore(text: str) -> None:
    assert "[CPF]" in redact(text)


def test_redact_keeps_digits_inside_identifiers() -> None:
    identifier = "550e8400-e29b-41d4-a716-44665544000a"

    assert redact(identifier) == identifier


def test_metrics_cover_text_completion_models() -> None:
    metrics = MetricsCallback()

    FakeListLLM(responses=["ok"]).invoke("oi", {"callbacks": [metrics]})

    assert metrics.metrics.llm_calls == 1
    assert metrics.metrics.llm_seconds > 0


def test_redact_preserves_dataclasses_and_messages() -> None:
    command = Command(update={"messages": [HumanMessage(f"cpf {CPF}")]}, goto="proximo")

    result = redact(command)

    assert result["goto"] == "proximo"
    [message] = result["update"]["messages"]
    assert isinstance(message, HumanMessage)
    assert message.text == "cpf [CPF]"


@pytest.fixture
def langsmith_runs() -> Iterator[dict[str, dict[str, Any]]]:
    """Roda o agente contra um LangSmith falso e devolve as runs recebidas, por id."""
    runs: dict[str, dict[str, Any]] = {}

    def handle(method: str, path: str, body: Any) -> tuple[int, Any]:
        if path.startswith("/runs") and method in ("POST", "PATCH"):
            runs.setdefault(body["id"], {}).update({k: v for k, v in body.items() if v is not None})
        return 200, {}

    with fake_server(handle) as url:
        client = langsmith_client(api_url=url, api_key="teste", auto_batch_tracing=False)
        tracer = langsmith_tracer(client, project="projeto-teste")
        run_agent({"callbacks": [tracer], "tags": ["lab"], "metadata": {"session_id": "sessao-1"}})
        tracer.wait_for_futures()
        yield runs


def test_langsmith_receives_the_run_tree(langsmith_runs: dict[str, dict[str, Any]]) -> None:
    runs = list(langsmith_runs.values())
    [root] = [r for r in runs if not r.get("parent_run_id")]
    llm = [r for r in runs if r["run_type"] == "llm"]
    [tool] = [r for r in runs if r["run_type"] == "tool"]

    assert root["name"] == "atendimento"
    assert root["session_name"] == "projeto-teste"
    assert "lab" in root["tags"]
    assert root["extra"]["metadata"]["session_id"] == "sessao-1"
    assert len(llm) == 2
    assert tool["name"] == "consultar_pedido"
    assert {r["trace_id"] for r in runs} == {root["id"]}
    assert all(r["parent_run_id"] in langsmith_runs for r in runs if r is not root)
    assert all(r["end_time"] for r in runs)


def test_langsmith_payload_has_no_pii(langsmith_runs: dict[str, dict[str, Any]]) -> None:
    payload = json.dumps(langsmith_runs, ensure_ascii=False)

    assert "[CPF]" in payload
    assert CPF not in payload
    assert EMAIL not in payload


@contextmanager
def capture_langfuse() -> Iterator[tuple[InMemorySpanExporter, RunnableConfig]]:
    """Cliente Langfuse que guarda os spans em memória em vez de enviá-los."""
    exporter = InMemorySpanExporter()
    # O SDK mantém um cliente por chave pública, por isso a chave é única.
    public_key = f"pk-{uuid.uuid4().hex}"
    client = langfuse_client(
        public_key=public_key,
        secret_key="sk-teste",
        base_url="http://127.0.0.1:9",
        span_exporter=exporter,
        tracer_provider=TracerProvider(),
    )
    try:
        yield exporter, {"callbacks": [langfuse_handler(public_key)]}
        client.flush()
    finally:
        client.shutdown()


@pytest.fixture
def langfuse_spans() -> list[ReadableSpan]:
    """Roda o agente e captura os spans que o Langfuse exportaria."""
    attrs = propagate_attributes(session_id="sessao-1", user_id="usuario-1", tags=["lab"])
    with capture_langfuse() as (exporter, config), attrs:
        run_agent(config)
    return list(exporter.get_finished_spans())


def attributes(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


def test_langfuse_exports_typed_observations(langfuse_spans: list[ReadableSpan]) -> None:
    by_type: dict[str, list[ReadableSpan]] = {}
    for span in langfuse_spans:
        by_type.setdefault(str(attributes(span)["langfuse.observation.type"]), []).append(span)

    assert len(by_type["generation"]) == 2
    assert [s.name for s in by_type["tool"]] == ["consultar_pedido"]
    usage = json.loads(attributes(by_type["generation"][0])["langfuse.observation.usage_details"])
    assert usage["total"] == usage["input"] + usage["output"] > 0
    model = attributes(by_type["generation"][0])["langfuse.observation.model.name"]
    assert model == "modelo-roteirizado"


def test_langfuse_spans_share_trace_and_session(langfuse_spans: list[ReadableSpan]) -> None:
    assert len({s.context.trace_id for s in langfuse_spans if s.context}) == 1
    for span in langfuse_spans:
        assert attributes(span)["session.id"] == "sessao-1"
        assert attributes(span)["user.id"] == "usuario-1"


def test_langfuse_payload_has_no_pii(langfuse_spans: list[ReadableSpan]) -> None:
    payload = json.dumps([attributes(s) for s in langfuse_spans], default=str, ensure_ascii=False)

    assert "[CPF]" in payload
    assert CPF not in payload
    assert EMAIL not in payload
    # Mascarar não pode significar perder o dado: o resto do conteúdo continua lá.
    assert "Pedido enviado. Aviso para [EMAIL]." in payload
    assert "fallback" not in payload.lower()


def failing_model() -> ScriptedChatModel:
    def fail(*_: Any) -> AIMessage:
        raise RuntimeError(f"cadastro inválido: CPF {CPF}, e-mail {EMAIL}")

    return ScriptedChatModel(responder=fail)


def test_langsmith_error_messages_have_no_pii() -> None:
    bodies: list[Any] = []

    def handle(method: str, path: str, body: Any) -> tuple[int, Any]:
        bodies.append(body)
        return 200, {}

    with fake_server(handle) as url:
        client = langsmith_client(api_url=url, api_key="teste", auto_batch_tracing=False)
        tracer = langsmith_tracer(client)
        with pytest.raises(RuntimeError):
            failing_model().invoke("oi", {"callbacks": [tracer]})
        tracer.wait_for_futures()

    payload = json.dumps(bodies, ensure_ascii=False)
    assert "cadastro inválido" in payload
    assert CPF not in payload
    assert EMAIL not in payload


def test_langfuse_error_messages_have_no_pii() -> None:
    with capture_langfuse() as (exporter, config), pytest.raises(RuntimeError):
        failing_model().invoke("oi", config)

    [span] = exporter.get_finished_spans()
    payload = json.dumps(
        [
            attributes(span),
            span.status.description,
            [dict(e.attributes or {}) for e in span.events],
        ],
        default=str,
        ensure_ascii=False,
    )
    assert "cadastro inválido" in payload
    assert CPF not in payload
    assert EMAIL not in payload


def test_langfuse_tool_error_status_has_no_pii() -> None:
    model = ScriptedChatModel.from_script(
        [tool_call("consultar_pedido", {"order_id": CPF}), AIMessage("Não encontrei.")]
    )

    with capture_langfuse() as (exporter, config):
        build_agent(model).invoke({"messages": [("human", "consulta")]}, config)

    [tool] = [s for s in exporter.get_finished_spans() if s.name == "consultar_pedido"]
    assert attributes(tool)["langfuse.observation.level"] == "ERROR"
    payload = json.dumps(
        [attributes(tool), tool.status.description], default=str, ensure_ascii=False
    )
    assert "[CPF]" in payload
    assert CPF not in payload
