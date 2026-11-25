"""Passo 3: o mesmo contrato verificado nos três provedores, com fixtures.

A partir daqui o código sob teste é o do laboratório, importado de ``labs``.

uv run pytest aulas/04-Testes-sem-Chave/03_cenarios_e_fixtures.py -v
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import anthropic
import httpx2
import openai
import pytest
from google import genai
from google.genai import types

from labs.llm_apis.anthropic_provider import AnthropicProvider
from labs.llm_apis.base import LLMProvider
from labs.llm_apis.gemini_provider import GeminiProvider
from labs.llm_apis.openai_provider import OpenAIProvider
from labs.llm_apis.types import AuthError

Reply = httpx2.Response


# --- Visto no passo 2, agora com uma classe base ----------------------------------


@dataclass
class Fake:
    queue: list[Reply] = field(default_factory=list)
    requests: int = 0
    unexpected: int = 0

    def handle(self, request: httpx2.Request) -> Reply:
        self.requests += 1
        if not self.queue:
            # Registrar antes de falhar: um SDK pode engolir a exceção.
            self.unexpected += 1
            raise AssertionError("requisição inesperada")
        return self.queue.pop(0)

    @property
    def http_client(self) -> httpx2.Client:
        return httpx2.Client(transport=httpx2.MockTransport(self.handle))

    def error(self, status: int) -> Reply:
        body = {"error": {"code": status, "message": "falha simulada"}}
        # retry-after-ms deixa as novas tentativas quase instantâneas.
        return Reply(status, json=body, headers={"retry-after-ms": "1"})

    def provider(self) -> LLMProvider:
        raise NotImplementedError

    def text(self, text: str) -> Reply:
        raise NotImplementedError

    def stream(self, *chunks: str) -> Reply:
        raise NotImplementedError


def sse(events: list[dict[str, Any]], *, named: bool = True) -> Reply:
    """Server-Sent Events: o formato do streaming."""
    frames = []
    for event in events:
        prefix = f"event: {event['type']}\n" if named else ""
        frames.append(f"{prefix}data: {json.dumps(event)}\n\n")
    return Reply(200, content="".join(frames), headers={"content-type": "text/event-stream"})


# --- Um simulador por provedor ----------------------------------------------------


class FakeOpenAI(Fake):
    def provider(self) -> LLMProvider:
        client = openai.OpenAI(api_key="teste", http_client=self.http_client)
        return OpenAIProvider(client, model="modelo")

    def text(self, text: str) -> Reply:
        part = {"type": "output_text", "text": text, "annotations": []}
        message = {"type": "message", "id": "m", "status": "completed", "role": "assistant"}
        return Reply(
            200,
            json={
                "id": "resp_1",
                "object": "response",
                "created_at": 0,
                "model": "modelo",
                "status": "completed",
                "output": [message | {"content": [part]}],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            },
        )

    def stream(self, *chunks: str) -> Reply:
        return sse(
            [
                {
                    "type": "response.output_text.delta",
                    "item_id": "m",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": chunk,
                    "sequence_number": index,
                    "logprobs": [],
                }
                for index, chunk in enumerate(chunks)
            ]
        )


class FakeAnthropic(Fake):
    def provider(self) -> LLMProvider:
        client = anthropic.Anthropic(api_key="teste", http_client=self.http_client)
        return AnthropicProvider(client, model="modelo")

    def _message(self, content: list[dict[str, Any]], stop: str | None) -> dict[str, Any]:
        return {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "modelo",
            "content": content,
            "stop_reason": stop,
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

    def text(self, text: str) -> Reply:
        return Reply(200, json=self._message([{"type": "text", "text": text}], "end_turn"))

    def stream(self, *chunks: str) -> Reply:
        block = {"type": "text", "text": ""}
        deltas = [
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": c}}
            for c in chunks
        ]
        return sse(
            [
                {"type": "message_start", "message": self._message([], None)},
                {"type": "content_block_start", "index": 0, "content_block": block},
                *deltas,
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 5},
                },
                {"type": "message_stop"},
            ]
        )


class FakeGemini(Fake):
    def provider(self) -> LLMProvider:
        retry = types.HttpRetryOptions(attempts=3, initial_delay=0.001, max_delay=0.002)
        options = types.HttpOptions(httpx_client=self.http_client, retry_options=retry)
        return GeminiProvider(genai.Client(api_key="teste", http_options=options), model="modelo")

    def _body(self, text: str) -> dict[str, Any]:
        content = {"role": "model", "parts": [{"text": text}]}
        return {"candidates": [{"content": content, "finishReason": "STOP"}]}

    def text(self, text: str) -> Reply:
        return Reply(200, json=self._body(text))

    def stream(self, *chunks: str) -> Reply:
        return sse([self._body(chunk) for chunk in chunks], named=False)


# --- Novo neste passo: fixtures ---------------------------------------------------


@pytest.fixture(params=[FakeOpenAI, FakeAnthropic, FakeGemini])
def fake(request: pytest.FixtureRequest) -> Iterator[Fake]:
    """Cada teste que pede ``fake`` roda uma vez por provedor."""
    instance: Fake = request.param()
    yield instance
    # Depois do teste: tudo que foi programado precisa ter sido usado.
    assert not instance.unexpected, "houve requisição sem resposta programada"
    assert not instance.queue, "sobrou resposta programada"


def test_generate(fake: Fake) -> None:
    fake.queue = [fake.text("Olá!")]

    assert fake.provider().generate("oi").text == "Olá!"


def test_stream_keeps_order(fake: Fake) -> None:
    fake.queue = [fake.stream("Bom ", "dia", "!")]

    assert list(fake.provider().stream("oi")) == ["Bom ", "dia", "!"]


def test_rate_limit_is_retried(fake: Fake) -> None:
    fake.queue = [fake.error(429), fake.error(429), fake.text("agora foi")]

    assert fake.provider().generate("oi").text == "agora foi"
    assert fake.requests == 3


def test_auth_error_is_not_retried(fake: Fake) -> None:
    fake.queue = [fake.error(401)]

    with pytest.raises(AuthError):
        fake.provider().generate("oi")

    assert fake.requests == 1
