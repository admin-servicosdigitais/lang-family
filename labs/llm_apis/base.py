"""Interface que os três adaptadores implementam, e o roteador com fallback."""

from collections.abc import Iterator, Sequence
from typing import Protocol

from pydantic import BaseModel

from labs.llm_apis.types import ChatResult, LLMError, Tool


class LLMProvider(Protocol):
    name: str
    model: str

    def generate(self, prompt: str, *, system: str | None = None) -> ChatResult: ...

    def stream(self, prompt: str, *, system: str | None = None) -> Iterator[str]: ...

    def structured[T: BaseModel](
        self, prompt: str, schema: type[T], *, system: str | None = None
    ) -> T: ...

    def run_tools(
        self,
        prompt: str,
        tools: list[Tool],
        *,
        system: str | None = None,
        max_turns: int = 5,
    ) -> ChatResult: ...


class FallbackRouter:
    """Tenta os provedores em ordem, avançando só em falhas transitórias.

    Erros de autenticação ou de requisição sobem imediatamente: trocar de provedor
    esconderia um defeito de configuração ou de código.
    """

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
