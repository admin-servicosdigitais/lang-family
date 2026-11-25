"""Passo 4: fallback entre provedores, só quando a falha é transitória."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

# --- Visto nos passos 1 a 3 (versão reduzida) -------------------------------------


@dataclass(frozen=True)
class ChatResult:
    text: str
    provider: str


class LLMError(Exception):
    retryable = False

    def __init__(self, provider: str, message: str) -> None:
        super().__init__(f"[{provider}] {message}")
        self.provider = provider


class AuthError(LLMError):
    pass


class RateLimitError(LLMError):
    retryable = True


class UnavailableError(LLMError):
    retryable = True


class LLMProvider(Protocol):
    name: str

    def generate(self, prompt: str, *, system: str | None = None) -> ChatResult: ...


class FakeProvider:
    def __init__(self, name: str, failure: LLMError | None = None) -> None:
        self.name = name
        self.failure = failure
        self.calls = 0

    def generate(self, prompt: str, *, system: str | None = None) -> ChatResult:
        self.calls += 1
        if self.failure:
            raise self.failure
        return ChatResult(f"resposta para: {prompt}", self.name)


# --- Novo neste passo -------------------------------------------------------------


class FallbackRouter:
    """Tenta os provedores em ordem, avançando só em falhas transitórias."""

    def __init__(self, providers: Sequence[LLMProvider]) -> None:
        if not providers:
            raise ValueError("informe ao menos um provedor")
        self.providers = list(providers)
        self.failures: list[LLMError] = []

    def generate(self, prompt: str, *, system: str | None = None) -> ChatResult:
        last_error: LLMError | None = None
        for provider in self.providers:
            try:
                return provider.generate(prompt, system=system)
            except LLMError as error:
                if not error.retryable:
                    raise
                self.failures.append(error)
                last_error = error
        assert last_error is not None
        raise last_error


def run(title: str, primary: FakeProvider, backup: FakeProvider) -> None:
    router = FallbackRouter([primary, backup])
    try:
        outcome = f"respondeu {router.generate('oi').provider}"
    except LLMError as error:
        outcome = f"erro {type(error).__name__} de {error.provider}"
    failures = [type(f).__name__ for f in router.failures]
    print(f"{title}: {outcome}; chamadas ao reserva={backup.calls}; falhas={failures}")


def main() -> None:
    run(
        "principal fora do ar",
        FakeProvider("openai", UnavailableError("openai", "503")),
        FakeProvider("anthropic"),
    )
    run(
        "chave errada",
        FakeProvider("openai", AuthError("openai", "401")),
        FakeProvider("anthropic"),
    )
    run(
        "todos com falha",
        FakeProvider("openai", UnavailableError("openai", "503")),
        FakeProvider("anthropic", RateLimitError("anthropic", "429")),
    )


if __name__ == "__main__":
    main()
