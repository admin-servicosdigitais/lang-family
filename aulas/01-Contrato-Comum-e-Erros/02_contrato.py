"""Passo 2: um contrato (Protocol) que qualquer provedor pode cumprir."""

from dataclasses import dataclass, field
from typing import Protocol

# --- Visto no passo 1 -------------------------------------------------------------


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class ChatResult:
    text: str
    provider: str
    model: str
    usage: Usage = field(default_factory=Usage)


# --- Novo neste passo -------------------------------------------------------------


class LLMProvider(Protocol):
    name: str
    model: str

    def generate(self, prompt: str, *, system: str | None = None) -> ChatResult: ...


# Nenhuma das classes abaixo herda de LLMProvider: basta ter o mesmo formato.
class FakeOpenAI:
    name = "openai"

    def __init__(self, model: str = "modelo-openai") -> None:
        self.model = model

    def generate(self, prompt: str, *, system: str | None = None) -> ChatResult:
        return ChatResult(f"Resposta da OpenAI para: {prompt}", self.name, self.model)


class FakeAnthropic:
    name = "anthropic"

    def __init__(self, model: str = "modelo-claude") -> None:
        self.model = model

    def generate(self, prompt: str, *, system: str | None = None) -> ChatResult:
        tone = f" ({system})" if system else ""
        return ChatResult(f"Resposta do Claude{tone} para: {prompt}", self.name, self.model)


class ProviderWithoutGenerate:
    name = "incompleto"
    model = "nenhum"


def ask(provider: LLMProvider, prompt: str) -> str:
    """Código de negócio: depende só do contrato, não de um SDK."""
    result = provider.generate(prompt, system="seja breve")
    return f"[{result.provider}/{result.model}] {result.text}"


def main() -> None:
    for provider in (FakeOpenAI(), FakeAnthropic()):
        print(ask(provider, "Cadê meu pedido?"))

    # O mypy rejeita esta chamada antes de executar. O "ignore" existe só para a
    # demonstração: sem ele, a checagem de tipos falha.
    try:
        ask(ProviderWithoutGenerate(), "oi")  # type: ignore[arg-type]
    except AttributeError as error:
        print(f"em execução, a falha só aparece na chamada: {error}")


if __name__ == "__main__":
    main()
