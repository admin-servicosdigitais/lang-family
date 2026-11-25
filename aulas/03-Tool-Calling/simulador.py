"""Simulador das três APIs, com respostas que pedem ferramentas.

É um arquivo de apoio, não um passo da aula. Os SDKs oficiais rodam sem alteração;
só o transporte é trocado. Os formatos são reproduções reduzidas de
``labs/llm_apis/tests/fakes.py`` e não comprovam o comportamento das APIs reais.
"""

import json
from dataclasses import dataclass, field
from itertools import count
from typing import Any

import httpx2

# (nome da ferramenta, argumentos). Texto no lugar do dicionário simula, na OpenAI,
# um JSON inválido.
Call = tuple[str, dict[str, Any] | str]
Reply = dict[str, Any]

_call_ids = count(1)
_response_ids = count(1)


@dataclass
class Simulator:
    """Entrega as respostas na ordem dada e registra o que o SDK enviou."""

    queue: list[Reply]
    requests: list[dict[str, Any]] = field(default_factory=list)

    @property
    def http_client(self) -> httpx2.Client:
        return httpx2.Client(transport=httpx2.MockTransport(self._handle))

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.read() or b"{}"))
        assert self.queue, f"requisição inesperada: {request.url.path}"
        return httpx2.Response(200, json=self.queue.pop(0))


def simulate(*replies: Reply) -> Simulator:
    return Simulator(list(replies))


# --- OpenAI (Responses API) -------------------------------------------------------


def _openai_response(output: list[dict[str, Any]]) -> Reply:
    return {
        "id": f"resp_{next(_response_ids)}",
        "object": "response",
        "created_at": 0,
        "model": "modelo-openai",
        "status": "completed",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 12,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": 4,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 16,
        },
    }


def openai_text(text: str) -> Reply:
    part = {"type": "output_text", "text": text, "annotations": []}
    message = {
        "type": "message",
        "id": "msg_1",
        "status": "completed",
        "role": "assistant",
        "content": [part],
    }
    return _openai_response([message])


def openai_tool_calls(*calls: Call) -> Reply:
    items = []
    for name, arguments in calls:
        number = next(_call_ids)
        raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
        items.append(
            {
                "type": "function_call",
                "id": f"fc_{number}",
                "call_id": f"call_{number}",
                "name": name,
                "arguments": raw,
                "status": "completed",
            }
        )
    return _openai_response(items)


# --- Claude (Messages API) --------------------------------------------------------


def _anthropic_message(content: list[dict[str, Any]], stop_reason: str) -> Reply:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "modelo-claude",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 14, "output_tokens": 6},
    }


def anthropic_text(text: str) -> Reply:
    return _anthropic_message([{"type": "text", "text": text}], "end_turn")


def anthropic_tool_calls(*calls: Call) -> Reply:
    thinking = {
        "type": "thinking",
        "thinking": "Preciso consultar os dados.",
        "signature": "assinatura-simulada",
    }
    blocks = [
        {"type": "tool_use", "id": f"toolu_{next(_call_ids)}", "name": name, "input": arguments}
        for name, arguments in calls
    ]
    return _anthropic_message([thinking, *blocks], "tool_use")


# --- Gemini (generateContent) -----------------------------------------------------


def _gemini_response(parts: list[dict[str, Any]]) -> Reply:
    return {
        "candidates": [
            {"content": {"role": "model", "parts": parts}, "finishReason": "STOP", "index": 0}
        ],
        "usageMetadata": {
            "promptTokenCount": 11,
            "candidatesTokenCount": 3,
            "thoughtsTokenCount": 20,
            "totalTokenCount": 34,
        },
        "modelVersion": "modelo-gemini",
    }


def gemini_text(text: str) -> Reply:
    return _gemini_response([{"text": text}])


def gemini_tool_calls(*calls: Call) -> Reply:
    parts: list[dict[str, Any]] = [
        {"functionCall": {"id": f"call_{next(_call_ids)}", "name": name, "args": arguments}}
        for name, arguments in calls
    ]
    # Em chamadas paralelas, a assinatura do raciocínio acompanha só a primeira.
    parts[0]["thoughtSignature"] = "YXNzaW5hdHVyYQ=="
    return _gemini_response(parts)
