"""Passo 3: erros normalizados, separados entre os que valem nova tentativa ou não."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

# --- Visto nos passos 1 e 2 (versão reduzida) -------------------------------------


@dataclass(frozen=True)
class ChatResult:
    text: str
    provider: str


# --- Novo neste passo: a hierarquia de erros --------------------------------------


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


# --- Novo neste passo: a tradução na fronteira com o SDK --------------------------


# Imitam as exceções que um SDK real lançaria.
class SDKStatusError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class SDKConnectionError(Exception):
    pass


class FakeProvider:
    name = "falso"

    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure

    @contextmanager
    def _errors(self) -> Iterator[None]:
        try:
            yield
        except SDKStatusError as error:
            raise error_for_status(self.name, error.status_code, str(error)) from error
        except SDKConnectionError as error:
            raise UnavailableError(self.name, str(error)) from error

    def generate(self, prompt: str) -> ChatResult:
        with self._errors():
            if self.failure:
                raise self.failure
            return ChatResult(f"resposta para: {prompt}", self.name)


def main() -> None:
    failures: list[Exception] = [
        SDKStatusError(401, "chave inválida"),
        SDKStatusError(400, "parâmetro desconhecido"),
        SDKStatusError(429, "limite de requisições"),
        SDKStatusError(503, "serviço indisponível"),
        SDKConnectionError("conexão recusada"),
    ]
    for failure in failures:
        try:
            FakeProvider(failure).generate("oi")
        except LLMError as error:
            cause = type(error.__cause__).__name__
            print(f"{type(error).__name__:<17} retryable={error.retryable!s:<5} {error} <- {cause}")


if __name__ == "__main__":
    main()
