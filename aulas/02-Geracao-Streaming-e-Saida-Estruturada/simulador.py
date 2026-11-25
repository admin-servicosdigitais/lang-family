"""Simulador das três APIs: respostas HTTP programadas no lugar da rede.

É um arquivo de apoio, não um passo da aula. Os SDKs oficiais rodam sem alteração;
só o transporte é trocado. Os formatos são reproduções reduzidas de
``labs/llm_apis/tests/fakes.py`` e não comprovam o comportamento das APIs reais.
Como a simulação funciona é o tema da aula 04.
"""

import json
from dataclasses import dataclass, field
from typing import Any

import httpx2


@dataclass(frozen=True)
class Reply:
    status: int = 200
    body: Any = None
    sse: str | None = None
    offline: bool = False


@dataclass
class Simulator:
    """Entrega as respostas na ordem dada e registra o que o SDK enviou."""

    queue: list[Reply]
    requests: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    @property
    def http_client(self) -> httpx2.Client:
        return httpx2.Client(transport=httpx2.MockTransport(self._handle))

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append((request.url.path, json.loads(request.read() or b"{}")))
        assert self.queue, f"requisição inesperada: {request.url.path}"
        reply = self.queue.pop(0)
        if reply.offline:
            raise httpx2.ConnectError("conexão recusada (simulada)", request=request)
        if reply.sse is not None:
            headers = {"content-type": "text/event-stream"}
            return httpx2.Response(200, content=reply.sse, headers=headers)
        # retry-after-ms deixa as novas tentativas dos SDKs quase instantâneas.
        return httpx2.Response(reply.status, json=reply.body, headers={"retry-after-ms": "1"})


def simulate(*replies: Reply) -> Simulator:
    return Simulator(list(replies))


def error(status: int) -> Reply:
    return Reply(status, {"error": {"code": status, "message": "falha simulada"}})


def offline() -> Reply:
    return Reply(offline=True)


def _sse(events: list[dict[str, Any]], *, named: bool = True) -> Reply:
    frames = []
    for event in events:
        prefix = f"event: {event['type']}\n" if named else ""
        frames.append(f"{prefix}data: {json.dumps(event)}\n\n")
    return Reply(sse="".join(frames))


# --- OpenAI (Responses API) -------------------------------------------------------


def _openai_response(part: dict[str, Any]) -> Reply:
    message = {
        "type": "message",
        "id": "msg_1",
        "status": "completed",
        "role": "assistant",
        "content": [part],
    }
    return Reply(
        body={
            "id": "resp_1",
            "object": "response",
            "created_at": 0,
            "model": "modelo-openai",
            "status": "completed",
            "output": [message],
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
    )


def openai_text(text: str) -> Reply:
    return _openai_response({"type": "output_text", "text": text, "annotations": []})


def openai_refusal() -> Reply:
    return _openai_response({"type": "refusal", "refusal": "não posso ajudar"})


def _openai_delta(kind: str, index: int, delta: str) -> dict[str, Any]:
    return {
        "type": kind,
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "delta": delta,
        "sequence_number": index,
        "logprobs": [],
    }


def openai_stream(*chunks: str) -> Reply:
    kind = "response.output_text.delta"
    return _sse([_openai_delta(kind, index, chunk) for index, chunk in enumerate(chunks)])


def openai_stream_refusal() -> Reply:
    return _sse([_openai_delta("response.refusal.delta", 0, "não posso ajudar")])


# --- Claude (Messages API) --------------------------------------------------------


def _anthropic_message(content: list[dict[str, Any]], stop_reason: str | None) -> dict[str, Any]:
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
    return Reply(body=_anthropic_message([{"type": "text", "text": text}], "end_turn"))


def anthropic_refusal() -> Reply:
    return Reply(body=_anthropic_message([], "refusal"))


def anthropic_stream(*chunks: str, stop_reason: str = "end_turn") -> Reply:
    deltas = [
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": chunk},
        }
        for chunk in chunks
    ]
    return _sse(
        [
            {"type": "message_start", "message": _anthropic_message([], None)},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            *deltas,
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": 6},
            },
            {"type": "message_stop"},
        ]
    )


def anthropic_stream_refusal() -> Reply:
    return anthropic_stream(stop_reason="refusal")


# --- Gemini (generateContent) -----------------------------------------------------


def _gemini_response(text: str) -> dict[str, Any]:
    return {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": text}]},
                "finishReason": "STOP",
                "index": 0,
            }
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
    return Reply(body=_gemini_response(text))


def _gemini_blocked() -> dict[str, Any]:
    return {"candidates": [{"finishReason": "SAFETY", "index": 0}]}


def gemini_refusal() -> Reply:
    return Reply(body=_gemini_blocked())


def gemini_stream(*chunks: str) -> Reply:
    return _sse([_gemini_response(chunk) for chunk in chunks], named=False)


def gemini_stream_refusal() -> Reply:
    return _sse([_gemini_blocked()], named=False)
