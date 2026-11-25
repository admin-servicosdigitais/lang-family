"""Passo 1: um dublê que cumpre o contrato testa a regra de negócio.

uv run pytest aulas/04-Testes-sem-Chave/01_duble_no_contrato.py -v
"""

from dataclasses import dataclass, field
from typing import Protocol

import pytest

# --- Visto na aula 01 (versão reduzida) -------------------------------------------


class LLMError(Exception):
    retryable = False


class AuthError(LLMError):
    pass


class UnavailableError(LLMError):
    retryable = True


class TextProvider(Protocol):
    def generate(self, prompt: str, *, system: str | None = None) -> str: ...


# --- Código sob teste: a regra de negócio -----------------------------------------

APOLOGY = "Estamos com instabilidade. Um atendente vai responder em breve."


def answer_customer(provider: TextProvider, question: str) -> str:
    try:
        return provider.generate(question, system="Seja breve.")
    except LLMError as error:
        if not error.retryable:
            raise  # chave errada é defeito nosso: não pode virar pedido de desculpas
        return APOLOGY


# --- Novo neste passo: o dublê ----------------------------------------------------


@dataclass
class FakeProvider:
    """Devolve o que o teste mandar e registra como foi chamado."""

    reply: str = "ok"
    failure: LLMError | None = None
    calls: list[tuple[str, str | None]] = field(default_factory=list)

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        self.calls.append((prompt, system))
        if self.failure:
            raise self.failure
        return self.reply


def test_returns_model_answer() -> None:
    provider = FakeProvider(reply="Seu pedido saiu.")

    assert answer_customer(provider, "Cadê meu pedido?") == "Seu pedido saiu."


def test_sends_system_instruction() -> None:
    provider = FakeProvider()

    answer_customer(provider, "oi")

    assert provider.calls == [("oi", "Seja breve.")]


def test_outage_becomes_apology() -> None:
    provider = FakeProvider(failure=UnavailableError("503"))

    assert answer_customer(provider, "oi") == APOLOGY


def test_configuration_error_is_not_hidden() -> None:
    provider = FakeProvider(failure=AuthError("401"))

    with pytest.raises(AuthError):
        answer_customer(provider, "oi")
