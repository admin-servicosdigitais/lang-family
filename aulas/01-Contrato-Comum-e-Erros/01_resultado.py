"""Passo 1: um resultado único para respostas de formatos diferentes."""

from dataclasses import FrozenInstanceError, dataclass, field
from typing import Any


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
class ChatResult:
    text: str
    provider: str
    model: str
    usage: Usage = field(default_factory=Usage)


# Respostas simuladas e simplificadas: cada API usa nomes próprios.
OPENAI_RAW: dict[str, Any] = {
    "model": "modelo-openai",
    "output_text": "Pedido enviado.",
    "usage": {"input_tokens": 12, "output_tokens": 4},
}
GEMINI_RAW: dict[str, Any] = {
    "model_version": "modelo-gemini",
    "text": "Pedido enviado.",
    "usage_metadata": {
        "prompt_token_count": 11,
        "candidates_token_count": 3,
        "thoughts_token_count": 20,
    },
}


def from_openai(raw: dict[str, Any]) -> ChatResult:
    usage = Usage(raw["usage"]["input_tokens"], raw["usage"]["output_tokens"])
    return ChatResult(raw["output_text"], "openai", raw["model"], usage)


def from_gemini(raw: dict[str, Any]) -> ChatResult:
    meta = raw["usage_metadata"]
    # Tokens de raciocínio são cobrados como saída.
    output = meta["candidates_token_count"] + meta["thoughts_token_count"]
    usage = Usage(meta["prompt_token_count"], output)
    return ChatResult(raw["text"], "gemini", raw["model_version"], usage)


def main() -> None:
    results = [from_openai(OPENAI_RAW), from_gemini(GEMINI_RAW)]

    # Daqui em diante o código não sabe de qual API cada resultado veio.
    total = Usage()
    for result in results:
        print(f"{result.provider}: {result.text} {result.usage}")
        total += result.usage
    print(f"total: {total}")

    try:
        results[0].text = "outro texto"  # type: ignore[misc]
    except FrozenInstanceError:
        print("ChatResult é imutável: a alteração foi bloqueada")


if __name__ == "__main__":
    main()
