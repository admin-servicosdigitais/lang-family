"""Adaptador do Claude usando a Messages API."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import anthropic
from anthropic.types import Message
from pydantic import BaseModel, ValidationError

from labs.llm_apis.types import (
    ChatResult,
    RefusalError,
    Tool,
    ToolLoopError,
    ToolRun,
    UnavailableError,
    Usage,
    error_for_status,
    run_tool,
)
from labs.shared.settings import model_for


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> None:
        self.client = client or anthropic.Anthropic()
        self.model = model or model_for(self.name)
        self.max_tokens = max_tokens

    @contextmanager
    def _errors(self) -> Iterator[None]:
        try:
            yield
        except anthropic.APIStatusError as error:
            raise error_for_status(self.name, error.status_code, str(error)) from error
        except anthropic.APIConnectionError as error:
            raise UnavailableError(self.name, str(error)) from error

    def _request(self, messages: list[Any], system: str | None) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if system:
            request["system"] = system
        return request

    def _check_refusal(self, message: Message) -> None:
        if message.stop_reason == "refusal":
            raise RefusalError(self.name, "o modelo recusou a requisição")

    @staticmethod
    def _usage(message: Message) -> Usage:
        return Usage(message.usage.input_tokens, message.usage.output_tokens)

    def _result(self, message: Message, usage: Usage, runs: list[ToolRun]) -> ChatResult:
        self._check_refusal(message)
        text = "".join(block.text for block in message.content if block.type == "text")
        return ChatResult(text, self.name, message.model, usage, tuple(runs))

    def generate(self, prompt: str, *, system: str | None = None) -> ChatResult:
        messages: list[Any] = [{"role": "user", "content": prompt}]
        with self._errors():
            message = self.client.messages.create(**self._request(messages, system))
        return self._result(message, self._usage(message), [])

    def stream(self, prompt: str, *, system: str | None = None) -> Iterator[str]:
        messages: list[Any] = [{"role": "user", "content": prompt}]
        with (
            self._errors(),
            self.client.messages.stream(**self._request(messages, system)) as stream,
        ):
            yield from stream.text_stream
            self._check_refusal(stream.get_final_message())

    def structured[T: BaseModel](
        self, prompt: str, schema: type[T], *, system: str | None = None
    ) -> T:
        messages: list[Any] = [{"role": "user", "content": prompt}]
        try:
            with self._errors():
                message = self.client.messages.parse(
                    **self._request(messages, system), output_format=schema
                )
        except ValidationError as error:
            # O SDK valida a resposta dentro de parse().
            raise RefusalError(self.name, "o modelo não devolveu o formato pedido") from error
        self._check_refusal(message)
        if message.parsed_output is None:
            raise RefusalError(self.name, "o modelo não devolveu o formato pedido")
        return message.parsed_output

    def run_tools(
        self,
        prompt: str,
        tools: list[Tool],
        *,
        system: str | None = None,
        max_turns: int = 5,
    ) -> ChatResult:
        definitions: list[Any] = [
            {"name": t.name, "description": t.description, "input_schema": t.json_schema}
            for t in tools
        ]
        messages: list[Any] = [{"role": "user", "content": prompt}]
        usage = Usage()
        runs: list[ToolRun] = []
        for _ in range(max_turns):
            with self._errors():
                message = self.client.messages.create(
                    **self._request(messages, system), tools=definitions
                )
            usage += self._usage(message)
            if message.stop_reason != "tool_use":
                return self._result(message, usage, runs)
            results = []
            for block in message.content:
                if block.type != "tool_use":
                    continue
                run = run_tool(tools, block.name, dict(block.input))
                runs.append(run)
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": run.output,
                        "is_error": run.is_error,
                    }
                )
            # Os blocos do assistente voltam intactos (inclusive os de raciocínio) e
            # todos os resultados vão em uma única mensagem de usuário.
            messages = [
                *messages,
                {"role": "assistant", "content": message.content},
                {"role": "user", "content": results},
            ]
        raise ToolLoopError(self.name, f"sem resposta final após {max_turns} turnos")
