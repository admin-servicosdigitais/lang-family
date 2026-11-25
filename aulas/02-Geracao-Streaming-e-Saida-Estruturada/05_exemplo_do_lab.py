"""Passo 5: os três adaptadores do laboratório, sob o mesmo contrato.

Os métodos das classes reproduzem ``labs/llm_apis/*_provider.py``. O ``run_tools``
ficou de fora porque é o tema da aula 03. O ``main`` é só para a aula.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol

import anthropic
import httpx2
import openai
from anthropic.types import Message
from google import genai
from google.genai import errors, types
from openai.types.responses import Response
from pydantic import BaseModel, ValidationError
from simulador import (
    anthropic_stream,
    anthropic_text,
    error,
    gemini_stream,
    gemini_text,
    openai_stream,
    openai_text,
    simulate,
)

# --- Do laboratório: shared/settings.py -------------------------------------------

DEFAULT_MODELS = {
    "openai": "gpt-6-luna",
    "anthropic": "claude-haiku-5-5",
    "gemini": "gemini-3.5-flash-lite",
}


def model_for(provider: str) -> str:
    return os.getenv(f"{provider.upper()}_MODEL") or DEFAULT_MODELS[provider]


# --- Visto na aula 01, reduzido ao que os adaptadores usam aqui -------------------


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class ToolRun:
    name: str
    arguments: dict[str, Any]
    output: str
    is_error: bool = False


@dataclass(frozen=True)
class ChatResult:
    text: str
    provider: str
    model: str
    usage: Usage = field(default_factory=Usage)
    tool_runs: tuple[ToolRun, ...] = ()


class LLMError(Exception):
    """Erro normalizado, independente do SDK que o gerou."""

    retryable = False

    def __init__(self, provider: str, message: str) -> None:
        super().__init__(f"[{provider}] {message}")
        self.provider = provider


class AuthError(LLMError):
    pass


class BadRequestError(LLMError):
    pass


class RateLimitError(LLMError):
    retryable = True


class UnavailableError(LLMError):
    """Erro 5xx, sobrecarga, timeout ou falha de conexão."""

    retryable = True


class RefusalError(LLMError):
    pass


def error_for_status(provider: str, status: int, message: str) -> LLMError:
    if status in (401, 403):
        return AuthError(provider, message)
    if status == 429:
        return RateLimitError(provider, message)
    if status >= 500 or status == 408:
        return UnavailableError(provider, message)
    return BadRequestError(provider, message)


class LLMProvider(Protocol):
    """O contrato do laboratório, ainda sem ``run_tools``."""

    name: str
    model: str

    def generate(self, prompt: str, *, system: str | None = None) -> ChatResult: ...

    def stream(self, prompt: str, *, system: str | None = None) -> Iterator[str]: ...

    def structured[T: BaseModel](
        self, prompt: str, schema: type[T], *, system: str | None = None
    ) -> T: ...


# --- Do laboratório: openai_provider.py -------------------------------------------


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
                    elif event.type == "response.refusal.delta":
                        raise RefusalError(self.name, "o modelo recusou a requisição")
                    elif event.type in ("response.failed", "error"):
                        raise LLMError(self.name, f"falha durante o stream: {event}")

    def structured[T: BaseModel](
        self, prompt: str, schema: type[T], *, system: str | None = None
    ) -> T:
        try:
            with self._errors():
                response = self.client.responses.parse(
                    **self._request(prompt, system), text_format=schema
                )
        except ValidationError as error:
            # O SDK valida a resposta dentro de parse().
            raise RefusalError(self.name, "o modelo não devolveu o formato pedido") from error
        parsed: T | None = response.output_parsed
        if parsed is None:
            raise RefusalError(self.name, "o modelo não devolveu o formato pedido")
        return parsed


# --- Do laboratório: anthropic_provider.py ----------------------------------------


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


# --- Do laboratório: gemini_provider.py -------------------------------------------

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


# --- Somente da aula: demonstração ------------------------------------------------


class Ticket(BaseModel):
    category: str
    priority: int


def use(provider: LLMProvider) -> None:
    """Código de negócio: o mesmo para qualquer provedor."""
    system = "Responda em uma frase."
    result = provider.generate("Cadê o meu pedido?", system=system)
    print(f"== {provider.name}")
    print(f"  generate  : {result.text} {result.usage}")
    print(f"  stream    : {list(provider.stream('Cadê o meu pedido?', system=system))}")
    print(f"  structured: {provider.structured('Classifique: pedido atrasado', Ticket)!r}")
    try:
        provider.generate("oi")
    except LLMError as exc:
        print(f"  chave ruim: {type(exc).__name__}")


def main() -> None:
    text = "Seu pedido foi enviado."
    pieces = ("Seu ", "pedido ", "saiu.")
    ticket = '{"category": "entrega", "priority": 2}'

    # Cada fila segue a ordem das quatro chamadas feitas em use().
    sim = simulate(openai_text(text), openai_stream(*pieces), openai_text(ticket), error(401))
    openai_client = openai.OpenAI(api_key="falsa", http_client=sim.http_client)
    use(OpenAIProvider(openai_client, model="modelo-openai"))

    sim = simulate(
        anthropic_text(text), anthropic_stream(*pieces), anthropic_text(ticket), error(401)
    )
    anthropic_client = anthropic.Anthropic(api_key="falsa", http_client=sim.http_client)
    use(AnthropicProvider(anthropic_client, model="modelo-claude"))

    sim = simulate(gemini_text(text), gemini_stream(*pieces), gemini_text(ticket), error(401))
    options = types.HttpOptions(httpx_client=sim.http_client)
    gemini_client = genai.Client(api_key="falsa", http_options=options)
    use(GeminiProvider(gemini_client, model="modelo-gemini"))


if __name__ == "__main__":
    main()
