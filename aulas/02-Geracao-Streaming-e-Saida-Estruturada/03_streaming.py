"""Passo 3: streaming, a resposta entregue em pedaços."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import anthropic
import openai
from google import genai
from google.genai import errors, types
from simulador import (
    Reply,
    Simulator,
    anthropic_stream,
    anthropic_stream_refusal,
    error,
    gemini_stream,
    gemini_stream_refusal,
    openai_stream,
    openai_stream_refusal,
    simulate,
)

PROMPT = "Cadê o meu pedido?"


# --- Visto nas aulas 01 e passo 2 (reduzido a erros de status) --------------------


class LLMError(Exception):
    def __init__(self, provider: str, message: str) -> None:
        super().__init__(f"[{provider}] {message}")
        self.provider = provider


class RefusalError(LLMError):
    pass


@contextmanager
def sdk_errors(provider: str) -> Iterator[None]:
    try:
        yield
    except (openai.APIStatusError, anthropic.APIStatusError, errors.APIError) as exc:
        raise LLMError(provider, f"falha da API: {type(exc).__name__}") from exc


# --- Novo neste passo -------------------------------------------------------------


def stream_openai(sim: Simulator) -> Iterator[str]:
    client = openai.OpenAI(api_key="falsa", http_client=sim.http_client, max_retries=0)
    with sdk_errors("openai"):
        events = client.responses.create(model="modelo-openai", input=PROMPT, stream=True)
        with events:  # fecha a conexão mesmo que o consumo seja interrompido
            for event in events:
                # Chegam eventos de vários tipos: é preciso escolher os que importam.
                if event.type == "response.output_text.delta":
                    yield event.delta
                elif event.type == "response.refusal.delta":
                    raise RefusalError("openai", "o modelo recusou a requisição")
                elif event.type in ("response.failed", "error"):
                    raise LLMError("openai", f"falha durante o stream: {event}")


def stream_anthropic(sim: Simulator) -> Iterator[str]:
    client = anthropic.Anthropic(api_key="falsa", http_client=sim.http_client, max_retries=0)
    with (
        sdk_errors("anthropic"),
        client.messages.stream(
            model="modelo-claude",
            max_tokens=1024,
            messages=[{"role": "user", "content": PROMPT}],
        ) as stream,
    ):
        yield from stream.text_stream
        # O motivo do término só é conhecido depois do último pedaço.
        if stream.get_final_message().stop_reason == "refusal":
            raise RefusalError("anthropic", "o modelo recusou a requisição")


def stream_gemini(sim: Simulator) -> Iterator[str]:
    options = types.HttpOptions(httpx_client=sim.http_client)
    client = genai.Client(api_key="falsa", http_options=options)
    with sdk_errors("gemini"):
        for chunk in client.models.generate_content_stream(model="modelo-gemini", contents=PROMPT):
            # Versão reduzida: o passo 5 traz a lista completa de motivos normais.
            for candidate in chunk.candidates or []:
                if candidate.finish_reason == types.FinishReason.SAFETY:
                    raise RefusalError("gemini", "resposta bloqueada")
            if chunk.text:
                yield chunk.text


def consume(title: str, stream: Callable[[Simulator], Iterator[str]], reply: Reply) -> None:
    sim = simulate(reply)
    chunks = stream(sim)
    before = len(sim.requests)
    received: list[str] = []
    try:
        for chunk in chunks:
            received.append(chunk)
        outcome = "fim"
    except LLMError as exc:
        outcome = f"{type(exc).__name__}"
    print(f"{title:<12} requisições antes/depois={before}/{len(sim.requests)} {received} {outcome}")


def main() -> None:
    pieces = ("Seu ", "pedido ", "saiu.")

    print("Texto em pedaços")
    consume("  openai", stream_openai, openai_stream(*pieces))
    consume("  anthropic", stream_anthropic, anthropic_stream(*pieces))
    consume("  gemini", stream_gemini, gemini_stream(*pieces))

    print("\nErro HTTP: só aparece quando o gerador é consumido")
    consume("  openai", stream_openai, error(429))
    consume("  anthropic", stream_anthropic, error(429))
    consume("  gemini", stream_gemini, error(429))

    print("\nRecusa no stream")
    consume("  openai", stream_openai, openai_stream_refusal())
    consume("  anthropic", stream_anthropic, anthropic_stream_refusal())
    consume("  gemini", stream_gemini, gemini_stream_refusal())

    print("\nRecusa depois de texto já entregue")
    reply = anthropic_stream("Seu ", "pedido ", stop_reason="refusal")
    consume("  anthropic", stream_anthropic, reply)


if __name__ == "__main__":
    main()
