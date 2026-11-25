"""Passo 2: trocar só o transporte HTTP, com o adaptador e o SDK rodando de verdade.

uv run pytest aulas/04-Testes-sem-Chave/02_simular_o_transporte.py -v
"""

import json
from dataclasses import dataclass, field
from typing import Any

import httpx2
import openai
import pytest

# --- Código sob teste: um adaptador reduzido (aula 02) ----------------------------


class UnavailableError(Exception):
    pass


def generate(client: openai.OpenAI, prompt: str, *, system: str) -> tuple[str, int]:
    try:
        response = client.responses.create(model="modelo", input=prompt, instructions=system)
    except (openai.InternalServerError, openai.APIConnectionError) as error:
        raise UnavailableError(str(error)) from error
    assert response.usage is not None
    return response.output_text, response.usage.output_tokens


# --- Novo neste passo: o simulador ------------------------------------------------


@dataclass
class FakeOpenAI:
    """Fila de respostas programadas e registro das requisições recebidas."""

    queue: list[httpx2.Response] = field(default_factory=list)
    requests: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append((request.url.path, json.loads(request.read())))
        return self.queue.pop(0)

    def client(self) -> openai.OpenAI:
        http_client = httpx2.Client(transport=httpx2.MockTransport(self.handle))
        return openai.OpenAI(api_key="teste", http_client=http_client, max_retries=0)


def text_response(text: str) -> httpx2.Response:
    """Reprodução reduzida do formato da Responses API."""
    part = {"type": "output_text", "text": text, "annotations": []}
    message = {"type": "message", "id": "m", "status": "completed", "role": "assistant"}
    body = {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": "modelo",
        "status": "completed",
        "output": [message | {"content": [part]}],
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
    return httpx2.Response(200, json=body)


def test_response_is_parsed_by_the_real_sdk() -> None:
    fake = FakeOpenAI(queue=[text_response("Olá!")])

    assert generate(fake.client(), "Diga olá", system="curto") == ("Olá!", 5)


def test_request_carries_prompt_and_system() -> None:
    fake = FakeOpenAI(queue=[text_response("ok")])

    generate(fake.client(), "Diga olá", system="curto")

    # Conferir o que saiu é tão importante quanto conferir o que voltou.
    path, body = fake.requests[0]
    assert path == "/v1/responses"
    assert body == {"model": "modelo", "input": "Diga olá", "instructions": "curto"}


def test_server_error_is_translated() -> None:
    fake = FakeOpenAI(queue=[httpx2.Response(503, json={"error": {"message": "fora do ar"}})])

    with pytest.raises(UnavailableError, match="fora do ar"):
        generate(fake.client(), "oi", system="curto")
