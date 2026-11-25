"""Contrato comum entre os provedores: resultados, ferramentas e erros."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError


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
            output = str(self.function(**args.model_dump()))
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
