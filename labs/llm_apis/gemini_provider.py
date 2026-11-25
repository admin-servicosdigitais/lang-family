"""Adaptador do Gemini usando o SDK ``google-genai``."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import httpx2
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

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

# Motivos de término que não indicam bloqueio nem falha. MAX_TOKENS e CONTINUATION
# são cortes por limite: o texto já entregue é válido, apenas incompleto.
_NORMAL_FINISH = (
    None,
    types.FinishReason.FINISH_REASON_UNSPECIFIED,
    types.FinishReason.STOP,
    types.FinishReason.MAX_TOKENS,
    types.FinishReason.CONTINUATION,
)


class GeminiProvider:
    name = "gemini"

    def __init__(self, client: genai.Client | None = None, model: str | None = None) -> None:
        # Diferente dos outros dois SDKs, o google-genai não repete requisições
        # por padrão: é preciso ligar as tentativas explicitamente.
        self.client = client or genai.Client(
            http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=3))
        )
        self.model = model or model_for(self.name)

    @contextmanager
    def _errors(self) -> Iterator[None]:
        try:
            yield
        except errors.APIError as error:
            raise error_for_status(self.name, error.code, str(error)) from error
        except httpx2.TransportError as error:
            raise UnavailableError(self.name, str(error)) from error

    @staticmethod
    def _usage(response: types.GenerateContentResponse) -> Usage:
        meta = response.usage_metadata
        if meta is None:
            return Usage()
        # Tokens de raciocínio são cobrados como saída.
        output = (meta.candidates_token_count or 0) + (meta.thoughts_token_count or 0)
        return Usage(meta.prompt_token_count or 0, output)

    def _parts(self, response: types.GenerateContentResponse) -> list[types.Part]:
        candidate = response.candidates[0] if response.candidates else None
        if candidate is None or candidate.content is None:
            reason = candidate.finish_reason if candidate else response.prompt_feedback
            raise RefusalError(self.name, f"resposta sem conteúdo: {reason}")
        return candidate.content.parts or []

    def _check_blocked(self, chunk: types.GenerateContentResponse) -> None:
        feedback = chunk.prompt_feedback
        if feedback and feedback.block_reason:
            raise RefusalError(self.name, f"requisição bloqueada: {feedback.block_reason}")
        for candidate in chunk.candidates or []:
            if candidate.finish_reason not in _NORMAL_FINISH:
                raise RefusalError(self.name, f"resposta interrompida: {candidate.finish_reason}")

    def _result(
        self, response: types.GenerateContentResponse, usage: Usage, runs: list[ToolRun]
    ) -> ChatResult:
        text = "".join(p.text for p in self._parts(response) if p.text and not p.thought)
        model = response.model_version or self.model
        return ChatResult(text, self.name, model, usage, tuple(runs))

    def generate(self, prompt: str, *, system: str | None = None) -> ChatResult:
        with self._errors():
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(system_instruction=system),
            )
        return self._result(response, self._usage(response), [])

    def stream(self, prompt: str, *, system: str | None = None) -> Iterator[str]:
        with self._errors():
            chunks = self.client.models.generate_content_stream(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(system_instruction=system),
            )
            for chunk in chunks:
                self._check_blocked(chunk)
                if chunk.text:
                    yield chunk.text

    def structured[T: BaseModel](
        self, prompt: str, schema: type[T], *, system: str | None = None
    ) -> T:
        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,
        )
        with self._errors():
            response = self.client.models.generate_content(
                model=self.model, contents=prompt, config=config
            )
        if not isinstance(response.parsed, schema):
            raise RefusalError(self.name, "o modelo não devolveu o formato pedido")
        return response.parsed

    def run_tools(
        self,
        prompt: str,
        tools: list[Tool],
        *,
        system: str | None = None,
        max_turns: int = 5,
    ) -> ChatResult:
        declarations = [
            types.FunctionDeclaration(
                name=t.name, description=t.description, parameters_json_schema=t.json_schema
            )
            for t in tools
        ]
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=declarations)],
            # O laço é controlado aqui, como nos outros provedores.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        contents: list[Any] = [types.Content(role="user", parts=[types.Part(text=prompt)])]
        usage = Usage()
        runs: list[ToolRun] = []
        for _ in range(max_turns):
            with self._errors():
                response = self.client.models.generate_content(
                    model=self.model, contents=contents, config=config
                )
            usage += self._usage(response)
            parts = self._parts(response)
            calls = [part.function_call for part in parts if part.function_call]
            if not calls:
                return self._result(response, usage, runs)
            replies = []
            for call in calls:
                run = run_tool(tools, call.name or "", dict(call.args or {}))
                runs.append(run)
                key = "error" if run.is_error else "result"
                reply = types.Part.from_function_response(name=run.name, response={key: run.output})
                assert reply.function_response is not None
                reply.function_response.id = call.id
                replies.append(reply)
            # O conteúdo do modelo volta intacto para preservar as thought signatures.
            contents += [
                types.Content(role="model", parts=parts),
                types.Content(role="user", parts=replies),
            ]
        raise ToolLoopError(self.name, f"sem resposta final após {max_turns} turnos")
