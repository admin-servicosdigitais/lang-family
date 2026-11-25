"""Passo 5: o ``run_tools`` do laboratório nos três adaptadores.

As classes reproduzem ``labs/llm_apis/``, só com o que o laço de ferramentas usa:
``generate``, ``stream`` e ``structured`` foram vistos na aula 02. O trecho final é
só para a aula.
"""

import json
import os
from collections.abc import Callable, Iterator
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
    anthropic_text,
    anthropic_tool_calls,
    gemini_text,
    gemini_tool_calls,
    openai_text,
    openai_tool_calls,
    simulate,
)

# --- Do laboratório: shared/settings.py ---------------------------------------------

DEFAULT_MODELS = {
    "openai": "gpt-6-luna",
    "anthropic": "claude-haiku-5-5",
    "gemini": "gemini-3.5-flash-lite",
}


def model_for(provider: str) -> str:
    return os.getenv(f"{provider.upper()}_MODEL") or DEFAULT_MODELS[provider]


# --- Do laboratório: types.py -------------------------------------------------------


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
        )


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


@dataclass(frozen=True)
class Tool:
    """Função Python exposta ao modelo; os argumentos são validados por Pydantic."""

    name: str
    description: str
    args_model: type[BaseModel]
    function: Callable[..., object]

    @property
    def json_schema(self) -> dict[str, Any]:
        return self.args_model.model_json_schema()

    def run(self, arguments: dict[str, Any]) -> ToolRun:
        """Executa a ferramenta. Falhas voltam ao modelo em vez de derrubar o fluxo."""
        try:
            args = self.args_model.model_validate(arguments)
            # dict(args) preserva modelos aninhados; model_dump() os viraria em dicts.
            output = str(self.function(**dict(args)))
        except ValidationError as error:
            return ToolRun(self.name, arguments, f"argumentos inválidos: {error}", True)
        except Exception as error:
            return ToolRun(self.name, arguments, f"{type(error).__name__}: {error}", True)
        return ToolRun(self.name, arguments, output)


def run_tool(tools: list[Tool], name: str, arguments: dict[str, Any]) -> ToolRun:
    for tool in tools:
        if tool.name == name:
            return tool.run(arguments)
    return ToolRun(name, arguments, f"ferramenta desconhecida: {name}", True)


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


class ToolLoopError(LLMError):
    pass


def error_for_status(provider: str, status: int, message: str) -> LLMError:
    if status in (401, 403):
        return AuthError(provider, message)
    if status == 429:
        return RateLimitError(provider, message)
    if status >= 500 or status == 408:
        return UnavailableError(provider, message)
    return BadRequestError(provider, message)


# --- Do laboratório: openai_provider.py ---------------------------------------------


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

    @staticmethod
    def _run_call(tools: list[Tool], name: str, raw_arguments: str) -> ToolRun:
        # A OpenAI envia os argumentos como texto, que pode não ser um JSON válido.
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as error:
            return ToolRun(name, {}, f"argumentos inválidos: {error}", True)
        return run_tool(tools, name, arguments)

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
                run = self._run_call(tools, call.name, call.arguments)
                runs.append(run)
                outputs.append(
                    {"type": "function_call_output", "call_id": call.call_id, "output": run.output}
                )
            # A conversa continua a partir da resposta anterior, guardada pela API.
            request = request | {"input": outputs, "previous_response_id": response.id}
        raise ToolLoopError(self.name, f"sem resposta final após {max_turns} turnos")


# --- Do laboratório: anthropic_provider.py ------------------------------------------


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


# --- Do laboratório: gemini_provider.py ---------------------------------------------


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

    def _result(
        self, response: types.GenerateContentResponse, usage: Usage, runs: list[ToolRun]
    ) -> ChatResult:
        text = "".join(p.text for p in self._parts(response) if p.text and not p.thought)
        model = response.model_version or self.model
        return ChatResult(text, self.name, model, usage, tuple(runs))

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


# --- Somente da aula: demonstração --------------------------------------------------


class ToolProvider(Protocol):
    """A parte do contrato usada nesta aula."""

    name: str

    def run_tools(
        self,
        prompt: str,
        tools: list[Tool],
        *,
        system: str | None = None,
        max_turns: int = 5,
    ) -> ChatResult: ...


class OrderArgs(BaseModel):
    order_id: str


def order_status(order_id: str) -> str:
    return {"P-1001": "enviado", "P-1002": "aguardando pagamento"}[order_id]


def use(provider: ToolProvider) -> None:
    """Código de negócio: o mesmo para qualquer provedor."""
    tool = Tool("consultar_pedido", "Consulta um pedido.", OrderArgs, order_status)
    result = provider.run_tools("Status do P-1001 e do P-9999?", [tool], system="Seja breve.")
    print(f"== {provider.name}")
    for run in result.tool_runs:
        print(f"  {run.name} {run.arguments} is_error={run.is_error} -> {run.output}")
    print(f"  resposta: {result.text}")
    print(f"  {result.usage}")


def main() -> None:
    calls = (
        ("consultar_pedido", {"order_id": "P-1001"}),
        ("consultar_pedido", {"order_id": "P-9999"}),
    )
    answer = "O P-1001 foi enviado; não encontrei o P-9999."

    sim = simulate(openai_tool_calls(*calls), openai_text(answer))
    openai_client = openai.OpenAI(api_key="falsa", http_client=sim.http_client)
    use(OpenAIProvider(openai_client, model="modelo-openai"))

    sim = simulate(anthropic_tool_calls(*calls), anthropic_text(answer))
    anthropic_client = anthropic.Anthropic(api_key="falsa", http_client=sim.http_client)
    use(AnthropicProvider(anthropic_client, model="modelo-claude"))

    sim = simulate(gemini_tool_calls(*calls), gemini_text(answer))
    options = types.HttpOptions(httpx_client=sim.http_client)
    gemini_client = genai.Client(api_key="falsa", http_options=options)
    use(GeminiProvider(gemini_client, model="modelo-gemini"))


if __name__ == "__main__":
    main()
