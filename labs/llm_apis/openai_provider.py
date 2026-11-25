"""Adaptador da OpenAI usando a Responses API."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import openai
from openai.types.responses import Response
from pydantic import BaseModel

from labs.llm_apis.types import (
    ChatResult,
    LLMError,
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


class OpenAIProvider:
    name = "openai"

    def __init__(self, client: openai.OpenAI | None = None, model: str | None = None) -> None:
        self.client = client or openai.OpenAI()
        self.model = model or model_for(self.name)

    @contextmanager
    def _errors(self) -> Iterator[None]:
        try:
            yield
        except openai.APIStatusError as error:
            raise error_for_status(self.name, error.status_code, str(error)) from error
        except openai.APIConnectionError as error:
            raise UnavailableError(self.name, str(error)) from error

    def _request(self, input_: Any, system: str | None) -> dict[str, Any]:
        request: dict[str, Any] = {"model": self.model, "input": input_}
        if system:
            request["instructions"] = system
        return request

    def _result(self, response: Response, usage: Usage, runs: list[ToolRun]) -> ChatResult:
        for item in response.output:
            if item.type == "message":
                for part in item.content:
                    if part.type == "refusal":
                        raise RefusalError(self.name, part.refusal)
        return ChatResult(response.output_text, self.name, response.model, usage, tuple(runs))

    @staticmethod
    def _usage(response: Response) -> Usage:
        if response.usage is None:
            return Usage()
        return Usage(response.usage.input_tokens, response.usage.output_tokens)

    def generate(self, prompt: str, *, system: str | None = None) -> ChatResult:
        with self._errors():
            response = self.client.responses.create(**self._request(prompt, system))
        return self._result(response, self._usage(response), [])

    def stream(self, prompt: str, *, system: str | None = None) -> Iterator[str]:
        with self._errors():
            events = self.client.responses.create(**self._request(prompt, system), stream=True)
            with events:
                for event in events:
                    if event.type == "response.output_text.delta":
                        yield event.delta
                    elif event.type in ("response.failed", "error"):
                        raise LLMError(self.name, f"falha durante o stream: {event}")

    def structured[T: BaseModel](
        self, prompt: str, schema: type[T], *, system: str | None = None
    ) -> T:
        with self._errors():
            response = self.client.responses.parse(
                **self._request(prompt, system), text_format=schema
            )
        parsed: T | None = response.output_parsed
        if parsed is None:
            raise RefusalError(self.name, "o modelo não devolveu o formato pedido")
        return parsed

    def run_tools(
        self,
        prompt: str,
        tools: list[Tool],
        *,
        system: str | None = None,
        max_turns: int = 5,
    ) -> ChatResult:
        definitions: Any = [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.json_schema,
                # O schema do Pydantic nem sempre cumpre as regras do modo estrito;
                # a validação dos argumentos é feita localmente em Tool.run.
                "strict": False,
            }
            for tool in tools
        ]
        request = self._request(prompt, system) | {"tools": definitions}
        usage = Usage()
        runs: list[ToolRun] = []
        for _ in range(max_turns):
            with self._errors():
                response = self.client.responses.create(**request)
            usage += self._usage(response)
            calls = [item for item in response.output if item.type == "function_call"]
            if not calls:
                return self._result(response, usage, runs)
            outputs = []
            for call in calls:
                run = run_tool(tools, call.name, json.loads(call.arguments))
                runs.append(run)
                outputs.append(
                    {"type": "function_call_output", "call_id": call.call_id, "output": run.output}
                )
            # A conversa continua a partir da resposta anterior, guardada pela API.
            request = request | {"input": outputs, "previous_response_id": response.id}
        raise ToolLoopError(self.name, f"sem resposta final após {max_turns} turnos")
