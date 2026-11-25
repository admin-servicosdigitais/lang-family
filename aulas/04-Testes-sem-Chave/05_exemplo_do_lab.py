"""Passo 5: a estrutura de testes do laboratório, com o simulador da OpenAI.

Os blocos "do laboratório" reproduzem ``labs/llm_apis/tests/``. Lá existem também os
simuladores do Claude e do Gemini. O código sob teste é importado de ``labs``.

uv run pytest aulas/04-Testes-sem-Chave/05_exemplo_do_lab.py -v
"""

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx2
import openai
import pytest
from pydantic import BaseModel, Field

from labs.llm_apis.base import LLMProvider
from labs.llm_apis.openai_provider import OpenAIProvider
from labs.llm_apis.types import AuthError, RefusalError, Tool, Usage
from labs.shared.settings import require_env

# --- Do laboratório: tests/fakes.py -------------------------------------------------


@dataclass
class Reply:
    status: int = 200
    body: Any = None
    sse: str | None = None


# Simuladores criados no teste atual; a fixture em conftest.py os confere ao final.
CREATED: list["FakeAPI"] = []


@dataclass
class FakeAPI(ABC):
    """Fila de respostas programadas + registro das requisições recebidas."""

    queue: list[Reply] = field(default_factory=list)
    requests: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        CREATED.append(self)

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.read() or b"{}")
        self.requests.append((request.url.path, body))
        if not self.queue:
            # Registrar é necessário: um SDK pode engolir a exceção abaixo.
            self.unexpected.append(request.url.path)
            raise AssertionError(f"requisição inesperada: {request.url.path}")
        reply = self.queue.pop(0)
        if reply.sse is not None:
            return httpx2.Response(
                200, content=reply.sse, headers={"content-type": "text/event-stream"}
            )
        # retry-after-ms mantém as novas tentativas dos SDKs quase instantâneas.
        return httpx2.Response(reply.status, json=reply.body, headers={"retry-after-ms": "1"})

    @property
    def http_client(self) -> httpx2.Client:
        return httpx2.Client(transport=httpx2.MockTransport(self.handle))

    def error(self, status: int) -> Reply:
        return Reply(status, {"error": {"code": status, "message": "falha simulada"}})

    @abstractmethod
    def provider(self, max_retries: int = 2) -> LLMProvider: ...

    @abstractmethod
    def text(self, text: str) -> Reply: ...

    def tool_call(self, name: str, arguments: dict[str, Any]) -> Reply:
        return self.tool_calls([(name, arguments)])

    @abstractmethod
    def tool_calls(self, calls: list[tuple[str, dict[str, Any]]]) -> Reply:
        """Uma resposta pedindo várias ferramentas, com ids call_1, call_2..."""

    @abstractmethod
    def stream(self, chunks: list[str]) -> Reply: ...

    @abstractmethod
    def refusal(self) -> Reply: ...

    @abstractmethod
    def stream_refusal(self) -> Reply: ...

    @abstractmethod
    def system_of(self, body: dict[str, Any]) -> str: ...

    @abstractmethod
    def tool_results_of(self, body: dict[str, Any]) -> list[str]: ...

    @abstractmethod
    def tool_result_ids_of(self, body: dict[str, Any]) -> list[str]: ...


def assert_all_used(fakes: list[FakeAPI]) -> None:
    """Falha se houve requisição sem resposta programada ou resposta que sobrou."""
    problems = [f"requisição inesperada: {path}" for fake in fakes for path in fake.unexpected]
    unused = sum(len(fake.queue) for fake in fakes)
    if unused:
        problems.append(f"{unused} resposta programada e não usada")
    assert not problems, "; ".join(problems)


def _sse(events: list[dict[str, Any]], *, named: bool = True) -> str:
    frames = []
    for event in events:
        prefix = f"event: {event['type']}\n" if named else ""
        frames.append(f"{prefix}data: {json.dumps(event)}\n\n")
    return "".join(frames)


class FakeOpenAI(FakeAPI):
    def provider(self, max_retries: int = 2) -> OpenAIProvider:
        client = openai.OpenAI(
            api_key="teste", http_client=self.http_client, max_retries=max_retries
        )
        return OpenAIProvider(client, model="gpt-teste")

    def _response(self, output: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "id": f"resp_{len(self.requests) + len(self.queue)}",
            "object": "response",
            "created_at": 0,
            "model": "gpt-teste",
            "status": "completed",
            "output": output,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": 10,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens": 5,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 15,
            },
        }

    def _message(self, part: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "message",
            "id": "msg_1",
            "status": "completed",
            "role": "assistant",
            "content": [part],
        }

    def text(self, text: str) -> Reply:
        part = {"type": "output_text", "text": text, "annotations": []}
        return Reply(body=self._response([self._message(part)]))

    def refusal(self) -> Reply:
        part = {"type": "refusal", "refusal": "não posso ajudar"}
        return Reply(body=self._response([self._message(part)]))

    def _function_call(self, number: int, name: str, arguments: str) -> dict[str, Any]:
        return {
            "type": "function_call",
            "id": f"fc_{number}",
            "call_id": f"call_{number}",
            "name": name,
            "arguments": arguments,
            "status": "completed",
        }

    def tool_calls(self, calls: list[tuple[str, dict[str, Any]]]) -> Reply:
        items = [
            self._function_call(number, name, json.dumps(arguments))
            for number, (name, arguments) in enumerate(calls, start=1)
        ]
        return Reply(body=self._response(items))

    def raw_tool_call(self, name: str, arguments: str) -> Reply:
        """Argumentos como texto livre, para simular um JSON inválido."""
        return Reply(body=self._response([self._function_call(1, name, arguments)]))

    def stream(self, chunks: list[str]) -> Reply:
        events = [
            {
                "type": "response.output_text.delta",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "delta": chunk,
                "sequence_number": index,
                "logprobs": [],
            }
            for index, chunk in enumerate(chunks)
        ]
        return Reply(sse=_sse(events))

    def stream_refusal(self) -> Reply:
        event = {
            "type": "response.refusal.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "não posso ajudar",
            "sequence_number": 0,
        }
        return Reply(sse=_sse([event]))

    def system_of(self, body: dict[str, Any]) -> str:
        return str(body["instructions"])

    def tool_results_of(self, body: dict[str, Any]) -> list[str]:
        return [i["output"] for i in body["input"] if i["type"] == "function_call_output"]

    def tool_result_ids_of(self, body: dict[str, Any]) -> list[str]:
        return [i["call_id"] for i in body["input"] if i["type"] == "function_call_output"]


# --- Do laboratório: tests/conftest.py ----------------------------------------------


@pytest.fixture(autouse=True)
def verify_fakes() -> Iterator[None]:
    """Todo simulador usado no teste precisa ter consumido exatamente o programado."""
    CREATED.clear()
    yield
    assert_all_used(CREATED)


class OrderArgs(BaseModel):
    order_id: str = Field(description="Número do pedido, ex.: P-1001")


ORDERS = {"P-1001": "enviado", "P-1002": "aguardando pagamento"}


@pytest.fixture
def order_tool() -> Tool:
    return Tool(
        name="consultar_pedido",
        description="Consulta o status de um pedido.",
        args_model=OrderArgs,
        function=lambda order_id: ORDERS[order_id],
    )


# --- Somente da aula ----------------------------------------------------------------


@pytest.fixture
def fake() -> FakeAPI:
    # No laboratório a fixture é parametrizada e devolve os três simuladores.
    return FakeOpenAI()


# --- Do laboratório: tests/test_providers.py ----------------------------------------


def test_generate_sends_system_and_prompt_and_reads_usage(fake: FakeAPI) -> None:
    fake.queue = [fake.text("Olá!")]

    result = fake.provider().generate("Diga olá", system="Responda em português")

    assert result.text == "Olá!"
    assert result.usage == Usage(10, 5)
    _, body = fake.requests[0]
    assert fake.system_of(body) == "Responda em português"
    assert "Diga olá" in json.dumps(body, ensure_ascii=False)


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


def test_rate_limit_is_retried_until_success(fake: FakeAPI) -> None:
    fake.queue = [fake.error(429), fake.error(429), fake.text("agora foi")]

    result = fake.provider(max_retries=2).generate("oi")

    assert result.text == "agora foi"
    assert len(fake.requests) == 3


def test_auth_error_is_not_retried(fake: FakeAPI) -> None:
    fake.queue = [fake.error(401)]

    with pytest.raises(AuthError):
        fake.provider(max_retries=2).generate("oi")

    assert len(fake.requests) == 1


def test_refusal_during_stream_becomes_a_typed_error(fake: FakeAPI) -> None:
    fake.queue = [fake.stream_refusal()]

    with pytest.raises(RefusalError):
        list(fake.provider().stream("pedido recusado"))


# --- Teste real, no padrão de tests/test_live.py ------------------------------------


@pytest.mark.live
def test_generate_against_the_real_api() -> None:
    """Pulado sem ``--live``. Com ``--live`` e a chave, gera custo."""
    require_env("OPENAI_API_KEY")

    result = OpenAIProvider().generate("Qual é a capital do Brasil? Responda só o nome.")

    assert "bras" in result.text.lower()
