"""O mesmo contrato verificado nos três provedores (offline)."""

import json
from contextlib import nullcontext

import pytest
from pydantic import BaseModel

from labs.llm_apis.base import FallbackRouter
from labs.llm_apis.tests.fakes import FakeAnthropic, FakeAPI, FakeGemini, FakeOpenAI
from labs.llm_apis.types import (
    AuthError,
    BadRequestError,
    RateLimitError,
    RefusalError,
    Tool,
    ToolLoopError,
    UnavailableError,
    Usage,
)


class Ticket(BaseModel):
    category: str
    priority: int


def test_generate_sends_system_and_prompt_and_reads_usage(fake: FakeAPI) -> None:
    fake.queue = [fake.text("Olá!")]

    result = fake.provider().generate("Diga olá", system="Responda em português")

    assert result.text == "Olá!"
    assert result.usage == Usage(10, 5)
    _, body = fake.requests[0]
    assert fake.system_of(body) == "Responda em português"
    assert "Diga olá" in json.dumps(body, ensure_ascii=False)


def test_stream_yields_chunks_in_order(fake: FakeAPI) -> None:
    fake.queue = [fake.stream(["Bom ", "dia", "!"])]

    chunks = list(fake.provider().stream("Cumprimente"))

    assert chunks == ["Bom ", "dia", "!"]


def test_structured_output_is_validated_into_model(fake: FakeAPI) -> None:
    fake.queue = [fake.text('{"category": "entrega", "priority": 2}')]

    ticket = fake.provider().structured("Classifique: pedido atrasado", Ticket)

    assert ticket == Ticket(category="entrega", priority=2)
    # O schema tem de ir na requisição; senão o formato não é garantido pela API.
    assert "priority" in json.dumps(fake.requests[0][1])


def test_tool_loop_executes_tool_and_returns_result_to_model(
    fake: FakeAPI, order_tool: Tool
) -> None:
    fake.queue = [
        fake.tool_call("consultar_pedido", {"order_id": "P-1001"}),
        fake.text("Seu pedido foi enviado."),
    ]

    result = fake.provider().run_tools("Cadê o pedido P-1001?", [order_tool])

    assert result.text == "Seu pedido foi enviado."
    assert [(r.name, r.output, r.is_error) for r in result.tool_runs] == [
        ("consultar_pedido", "enviado", False)
    ]
    assert fake.tool_results_of(fake.requests[1][1]) == ["enviado"]
    assert result.usage == Usage(20, 10)


@pytest.mark.parametrize(
    ("name", "arguments", "expected"),
    [
        ("consultar_pedido", {}, "argumentos inválidos"),
        ("consultar_pedido", {"order_id": "P-9999"}, "KeyError"),
        ("apagar_tudo", {}, "ferramenta desconhecida"),
    ],
)
def test_tool_failures_are_reported_to_the_model(
    fake: FakeAPI, order_tool: Tool, name: str, arguments: dict[str, str], expected: str
) -> None:
    fake.queue = [fake.tool_call(name, arguments), fake.text("Não consegui consultar.")]

    result = fake.provider().run_tools("consulta", [order_tool])

    assert result.tool_runs[0].is_error
    assert expected in fake.tool_results_of(fake.requests[1][1])[0]


def test_parallel_tool_calls_are_all_answered_in_one_turn(fake: FakeAPI, order_tool: Tool) -> None:
    calls = [("consultar_pedido", {"order_id": order}) for order in ("P-1001", "P-1002")]
    fake.queue = [fake.tool_calls(calls), fake.text("Um enviado, outro aguardando.")]

    result = fake.provider().run_tools("status dos dois pedidos", [order_tool])

    assert [r.output for r in result.tool_runs] == ["enviado", "aguardando pagamento"]
    assert len(fake.requests) == 2
    second = fake.requests[1][1]
    assert fake.tool_results_of(second) == ["enviado", "aguardando pagamento"]
    # Cada resultado precisa apontar para o pedido que o originou.
    assert fake.tool_result_ids_of(second) == ["call_1", "call_2"]


def test_openai_malformed_tool_arguments_are_reported_to_the_model(order_tool: Tool) -> None:
    fake = FakeOpenAI()
    fake.queue = [
        fake.raw_tool_call("consultar_pedido", '{"order_id": "P-10'),
        fake.text("Pode repetir o número?"),
    ]

    result = fake.provider().run_tools("status", [order_tool])

    assert result.text == "Pode repetir o número?"
    assert result.tool_runs[0].is_error
    assert "argumentos inválidos" in fake.tool_results_of(fake.requests[1][1])[0]


def test_tool_receives_nested_models_as_instances() -> None:
    class Item(BaseModel):
        sku: str

    class Args(BaseModel):
        items: list[Item]
        main: Item

    def describe(items: list[Item], main: Item) -> str:
        return f"{main.sku}: {', '.join(item.sku for item in items)}"

    tool = Tool("descrever", "Descreve itens.", Args, describe)

    run = tool.run({"items": [{"sku": "A1"}, {"sku": "B2"}], "main": {"sku": "A1"}})

    assert (run.is_error, run.output) == (False, "A1: A1, B2")


def test_tool_loop_stops_at_max_turns(fake: FakeAPI, order_tool: Tool) -> None:
    call = fake.tool_call("consultar_pedido", {"order_id": "P-1001"})
    fake.queue = [call, call]

    with pytest.raises(ToolLoopError):
        fake.provider().run_tools("consulta", [order_tool], max_turns=2)

    assert len(fake.requests) == 2


def test_refusal_becomes_a_typed_error(fake: FakeAPI) -> None:
    fake.queue = [fake.refusal()]

    with pytest.raises(RefusalError):
        fake.provider().generate("pedido recusado")


def test_rate_limit_is_retried_until_success(fake: FakeAPI) -> None:
    fake.queue = [fake.error(429), fake.error(429), fake.text("agora foi")]

    result = fake.provider(max_retries=2).generate("oi")

    assert result.text == "agora foi"
    assert len(fake.requests) == 3


@pytest.mark.parametrize(
    ("status", "error_type"),
    [(401, AuthError), (400, BadRequestError), (429, RateLimitError), (503, UnavailableError)],
)
def test_http_errors_are_normalized(
    fake: FakeAPI, status: int, error_type: type[Exception]
) -> None:
    fake.queue = [fake.error(status)]

    with pytest.raises(error_type):
        fake.provider(max_retries=0).generate("oi")


def test_auth_error_is_not_retried(fake: FakeAPI) -> None:
    fake.queue = [fake.error(401)]

    with pytest.raises(AuthError):
        fake.provider(max_retries=2).generate("oi")

    assert len(fake.requests) == 1


def test_router_falls_back_to_next_provider_on_outage() -> None:
    primary, backup = FakeOpenAI(), FakeAnthropic()
    primary.queue = [primary.error(503)]
    backup.queue = [backup.text("resposta do reserva")]
    router = FallbackRouter([primary.provider(max_retries=0), backup.provider()])

    result = router.generate("oi")

    assert (result.provider, result.text) == ("anthropic", "resposta do reserva")
    assert [f.provider for f in router.failures] == ["openai"]


def test_router_does_not_hide_configuration_errors() -> None:
    primary, backup = FakeGemini(), FakeOpenAI()
    primary.queue = [primary.error(401)]
    router = FallbackRouter([primary.provider(), backup.provider()])

    with pytest.raises(AuthError):
        router.generate("oi")

    assert backup.requests == []


def test_gemini_tool_loop_preserves_thought_signature(order_tool: Tool) -> None:
    fake = FakeGemini()
    fake.queue = [
        fake.tool_call("consultar_pedido", {"order_id": "P-1002"}),
        fake.text("Aguardando pagamento."),
    ]

    fake.provider().run_tools("status do P-1002", [order_tool])

    model_turn = fake.requests[1][1]["contents"][1]
    assert model_turn["parts"][0]["thoughtSignature"] == "YXNzaW5hdHVyYQ=="
    assert fake.requests[1][1]["contents"][2]["parts"][0]["functionResponse"]["id"] == "call_1"


@pytest.mark.parametrize(
    "reply",
    ['{"category": "ent', '{"category": "entrega"}'],
    ids=["json-incompleto", "campo-ausente"],
)
def test_invalid_structured_output_becomes_a_typed_error(fake: FakeAPI, reply: str) -> None:
    fake.queue = [fake.text(reply)]

    with pytest.raises(RefusalError):
        fake.provider().structured("Classifique", Ticket)


def test_refusal_during_stream_becomes_a_typed_error(fake: FakeAPI) -> None:
    fake.queue = [fake.stream_refusal()]

    with pytest.raises(RefusalError):
        list(fake.provider().stream("pedido recusado"))


def test_gemini_blocked_prompt_during_stream_becomes_a_typed_error() -> None:
    fake = FakeGemini()
    fake.queue = [fake.stream_events([{"promptFeedback": {"blockReason": "SAFETY"}}])]

    with pytest.raises(RefusalError):
        list(fake.provider().stream("pedido bloqueado"))


@pytest.mark.parametrize(("reason", "blocked"), [("SAFETY", True), ("MAX_TOKENS", False)])
def test_gemini_stream_finish_reason_after_partial_text(reason: str, blocked: bool) -> None:
    fake = FakeGemini()
    first = {"candidates": [{"content": {"role": "model", "parts": [{"text": "Bom "}]}}]}
    last = {"candidates": [{"finishReason": reason}]}
    fake.queue = [fake.stream_events([first, last])]
    chunks: list[str] = []

    with pytest.raises(RefusalError) if blocked else nullcontext():
        chunks.extend(fake.provider().stream("x"))

    # O texto já entregue não é desfeito: quem consome precisa tratar a falha no meio.
    assert chunks == ["Bom "]


def test_openai_tool_loop_chains_previous_response(order_tool: Tool) -> None:
    fake = FakeOpenAI()
    fake.queue = [
        fake.tool_call("consultar_pedido", {"order_id": "P-1001"}),
        fake.text("ok"),
    ]

    fake.provider().run_tools("status", [order_tool])

    assert fake.requests[1][1]["previous_response_id"].startswith("resp_")
