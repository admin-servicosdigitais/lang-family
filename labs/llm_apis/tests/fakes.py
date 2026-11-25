"""Simuladores das APIs da OpenAI, Anthropic e Gemini na camada HTTP.

Os SDKs oficiais são usados sem alteração: apenas o transporte é trocado. Os
formatos abaixo são reproduções para teste; a prova contra os serviços reais
está em ``test_live.py``.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import anthropic
import httpx2
import openai
from google import genai
from google.genai import types

from labs.llm_apis.anthropic_provider import AnthropicProvider
from labs.llm_apis.base import LLMProvider
from labs.llm_apis.gemini_provider import GeminiProvider
from labs.llm_apis.openai_provider import OpenAIProvider


@dataclass
class Reply:
    status: int = 200
    body: Any = None
    sse: str | None = None


@dataclass
class FakeAPI(ABC):
    """Fila de respostas programadas + registro das requisições recebidas."""

    queue: list[Reply] = field(default_factory=list)
    requests: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.read() or b"{}")
        self.requests.append((request.url.path, body))
        assert self.queue, f"requisição inesperada: {request.url.path}"
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


class FakeAnthropic(FakeAPI):
    def provider(self, max_retries: int = 2) -> AnthropicProvider:
        client = anthropic.Anthropic(
            api_key="teste", http_client=self.http_client, max_retries=max_retries
        )
        return AnthropicProvider(client, model="claude-teste")

    def _message(self, content: list[dict[str, Any]], stop_reason: str) -> dict[str, Any]:
        return {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-teste",
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

    def text(self, text: str) -> Reply:
        return Reply(body=self._message([{"type": "text", "text": text}], "end_turn"))

    def refusal(self) -> Reply:
        return Reply(body=self._message([], "refusal"))

    def tool_calls(self, calls: list[tuple[str, dict[str, Any]]]) -> Reply:
        blocks = [
            {"type": "tool_use", "id": f"call_{number}", "name": name, "input": arguments}
            for number, (name, arguments) in enumerate(calls, start=1)
        ]
        return Reply(body=self._message(blocks, "tool_use"))

    def stream_refusal(self) -> Reply:
        return self.stream([], stop_reason="refusal")

    def stream(self, chunks: list[str], stop_reason: str = "end_turn") -> Reply:
        start = self._message([], "end_turn") | {"stop_reason": None}
        events: list[dict[str, Any]] = [
            {"type": "message_start", "message": start},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            *(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": chunk},
                }
                for chunk in chunks
            ),
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": 5},
            },
            {"type": "message_stop"},
        ]
        return Reply(sse=_sse(events))

    def system_of(self, body: dict[str, Any]) -> str:
        return str(body["system"])

    def tool_results_of(self, body: dict[str, Any]) -> list[str]:
        blocks = body["messages"][-1]["content"]
        return [b["content"] for b in blocks if b["type"] == "tool_result"]

    def tool_result_ids_of(self, body: dict[str, Any]) -> list[str]:
        blocks = body["messages"][-1]["content"]
        return [b["tool_use_id"] for b in blocks if b["type"] == "tool_result"]


class FakeGemini(FakeAPI):
    def provider(self, max_retries: int = 2) -> GeminiProvider:
        retry = types.HttpRetryOptions(
            attempts=max_retries + 1, initial_delay=0.001, max_delay=0.002, jitter=0.001
        )
        options = types.HttpOptions(httpx_client=self.http_client, retry_options=retry)
        client = genai.Client(api_key="teste", http_options=options)
        return GeminiProvider(client, model="gemini-teste")

    def _response(self, parts: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "candidates": [
                {"content": {"role": "model", "parts": parts}, "finishReason": "STOP", "index": 0}
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            },
            "modelVersion": "gemini-teste",
        }

    def text(self, text: str) -> Reply:
        return Reply(body=self._response([{"text": text}]))

    def refusal(self) -> Reply:
        return Reply(body={"candidates": [{"finishReason": "SAFETY", "index": 0}]})

    def tool_calls(self, calls: list[tuple[str, dict[str, Any]]]) -> Reply:
        parts: list[dict[str, Any]] = [
            {"functionCall": {"id": f"call_{number}", "name": name, "args": arguments}}
            for number, (name, arguments) in enumerate(calls, start=1)
        ]
        # Em chamadas paralelas, a assinatura acompanha só a primeira.
        parts[0]["thoughtSignature"] = "YXNzaW5hdHVyYQ=="
        return Reply(body=self._response(parts))

    def stream(self, chunks: list[str]) -> Reply:
        return self.stream_events([self._response([{"text": chunk}]) for chunk in chunks])

    def stream_events(self, events: list[dict[str, Any]]) -> Reply:
        return Reply(sse=_sse(events, named=False))

    def stream_refusal(self) -> Reply:
        return self.stream_events([{"candidates": [{"finishReason": "SAFETY", "index": 0}]}])

    def system_of(self, body: dict[str, Any]) -> str:
        return str(body["systemInstruction"]["parts"][0]["text"])

    def tool_results_of(self, body: dict[str, Any]) -> list[str]:
        parts = body["contents"][-1]["parts"]
        return [str(next(iter(p["functionResponse"]["response"].values()))) for p in parts]

    def tool_result_ids_of(self, body: dict[str, Any]) -> list[str]:
        return [str(p["functionResponse"]["id"]) for p in body["contents"][-1]["parts"]]


FAKES: dict[str, type[FakeAPI]] = {
    "openai": FakeOpenAI,
    "anthropic": FakeAnthropic,
    "gemini": FakeGemini,
}
