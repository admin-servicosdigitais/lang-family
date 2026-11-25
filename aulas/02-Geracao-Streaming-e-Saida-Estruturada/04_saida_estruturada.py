"""Passo 4: saída estruturada, a resposta validada em um modelo Pydantic."""

import json
from collections.abc import Callable
from typing import Any

import anthropic
import openai
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError
from simulador import (
    Reply,
    Simulator,
    anthropic_refusal,
    anthropic_text,
    gemini_refusal,
    gemini_text,
    openai_refusal,
    openai_text,
    simulate,
)

PROMPT = "Classifique: meu pedido está atrasado."


# --- Visto na aula 01 -------------------------------------------------------------


class LLMError(Exception):
    def __init__(self, provider: str, message: str) -> None:
        super().__init__(f"[{provider}] {message}")
        self.provider = provider


class RefusalError(LLMError):
    pass


# --- Novo neste passo -------------------------------------------------------------


class Ticket(BaseModel):
    """Classificação de um chamado."""

    category: str
    priority: int = Field(ge=1, le=5)


NO_FORMAT = "o modelo não devolveu o formato pedido"


def structured_openai[T: BaseModel](sim: Simulator, schema: type[T]) -> T:
    client = openai.OpenAI(api_key="falsa", http_client=sim.http_client)
    try:
        response = client.responses.parse(model="modelo-openai", input=PROMPT, text_format=schema)
    except ValidationError as exc:
        raise RefusalError("openai", NO_FORMAT) from exc
    parsed: T | None = response.output_parsed
    if parsed is None:  # recusa: não há texto para validar
        raise RefusalError("openai", NO_FORMAT)
    return parsed


def structured_anthropic[T: BaseModel](sim: Simulator, schema: type[T]) -> T:
    client = anthropic.Anthropic(api_key="falsa", http_client=sim.http_client)
    try:
        message = client.messages.parse(
            model="modelo-claude",
            max_tokens=1024,
            messages=[{"role": "user", "content": PROMPT}],
            output_format=schema,
        )
    except ValidationError as exc:
        raise RefusalError("anthropic", NO_FORMAT) from exc
    if message.stop_reason == "refusal" or message.parsed_output is None:
        raise RefusalError("anthropic", NO_FORMAT)
    return message.parsed_output


def structured_gemini[T: BaseModel](sim: Simulator, schema: type[T]) -> T:
    options = types.HttpOptions(httpx_client=sim.http_client)
    client = genai.Client(api_key="falsa", http_options=options)
    config = types.GenerateContentConfig(
        response_mime_type="application/json", response_schema=schema
    )
    response = client.models.generate_content(model="modelo-gemini", contents=PROMPT, config=config)
    # Este SDK não lança exceção: quando a validação falha, parsed não é o modelo.
    if not isinstance(response.parsed, schema):
        raise RefusalError("gemini", NO_FORMAT)
    return response.parsed


type Structured = Callable[[Simulator, type[Ticket]], Ticket]

PROVIDERS: dict[str, tuple[Structured, Callable[[str], Reply], Callable[[], Reply]]] = {
    "openai": (structured_openai, openai_text, openai_refusal),
    "anthropic": (structured_anthropic, anthropic_text, anthropic_refusal),
    "gemini": (structured_gemini, gemini_text, gemini_refusal),
}


def schema_sent(body: dict[str, Any]) -> Any:
    """Cada API recebe o schema em um campo diferente."""
    if "text" in body:
        return {"text.format": body["text"]["format"]}
    if "output_config" in body:
        return {"output_config.format": body["output_config"]["format"]}
    config = body["generationConfig"]
    return {f"generationConfig.{k}": v for k, v in config.items()}


def main() -> None:
    valid = '{"category": "entrega", "priority": 2}'
    cases = {
        "JSON incompleto": '{"category": "entr',
        "campo ausente": '{"category": "entrega"}',
        "fora do limite": '{"category": "entrega", "priority": 9}',
    }

    for name, (structured, text, refusal) in PROVIDERS.items():
        print(f"== {name}")
        sim = simulate(text(valid))
        print(f"  válido: {structured(sim, Ticket)!r}")
        sent = schema_sent(sim.requests[0][1])
        print(f"  schema enviado: {json.dumps(sent, ensure_ascii=False)}")

        replies = {label: text(raw) for label, raw in cases.items()} | {"recusa": refusal()}
        for label, reply in replies.items():
            try:
                structured(simulate(reply), Ticket)
            except RefusalError as exc:
                cause = type(exc.__cause__).__name__ if exc.__cause__ else "sem exceção"
                print(f"  {label:<16} -> RefusalError (causa: {cause})")


if __name__ == "__main__":
    main()
