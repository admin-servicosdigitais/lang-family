"""Passo 2: traduzir os erros de cada SDK, detectar recusas e repetir quando vale a pena."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import anthropic
import httpx2
import openai
from google import genai
from google.genai import errors, types
from simulador import (
    Reply,
    Simulator,
    anthropic_refusal,
    anthropic_text,
    error,
    gemini_refusal,
    gemini_text,
    offline,
    openai_refusal,
    openai_text,
    simulate,
)

PROMPT = "Cadê o meu pedido?"


# --- Visto na aula 01 -------------------------------------------------------------


class LLMError(Exception):
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
    retryable = True


class RefusalError(LLMError):
    pass


def error_for_status(provider: str, status: int, message: str) -> LLMError:
    if status in (401, 403):
        return AuthError(provider, message)
    if status == 429:
        return RateLimitError(provider, message)
    if status >= 500 or status == 408:
        return UnavailableError(provider, message)
    return BadRequestError(provider, message)


# --- Novo neste passo: cada SDK tem as próprias exceções --------------------------


@contextmanager
def openai_errors() -> Iterator[None]:
    try:
        yield
    except openai.APIStatusError as exc:
        raise error_for_status("openai", exc.status_code, str(exc)) from exc
    except openai.APIConnectionError as exc:
        raise UnavailableError("openai", str(exc)) from exc


@contextmanager
def anthropic_errors() -> Iterator[None]:
    try:
        yield
    except anthropic.APIStatusError as exc:
        raise error_for_status("anthropic", exc.status_code, str(exc)) from exc
    except anthropic.APIConnectionError as exc:
        raise UnavailableError("anthropic", str(exc)) from exc


@contextmanager
def gemini_errors() -> Iterator[None]:
    try:
        yield
    except errors.APIError as exc:
        raise error_for_status("gemini", exc.code, str(exc)) from exc
    except httpx2.TransportError as exc:
        # O google-genai deixa a falha de rede subir como erro do transporte HTTP.
        raise UnavailableError("gemini", str(exc)) from exc


# --- Novo neste passo: recusa chega com status 200 --------------------------------


def generate_openai(sim: Simulator) -> str:
    client = openai.OpenAI(api_key="falsa", http_client=sim.http_client)
    with openai_errors():
        response = client.responses.create(model="modelo-openai", input=PROMPT)
    for item in response.output:
        if item.type == "message":
            for part in item.content:
                if part.type == "refusal":
                    raise RefusalError("openai", part.refusal)
    return response.output_text


def generate_anthropic(sim: Simulator) -> str:
    client = anthropic.Anthropic(api_key="falsa", http_client=sim.http_client)
    with anthropic_errors():
        message = client.messages.create(
            model="modelo-claude",
            max_tokens=1024,
            messages=[{"role": "user", "content": PROMPT}],
        )
    if message.stop_reason == "refusal":
        raise RefusalError("anthropic", "o modelo recusou a requisição")
    return "".join(block.text for block in message.content if block.type == "text")


def gemini_client(sim: Simulator, *, attempts: int | None) -> genai.Client:
    # Os intervalos minúsculos só servem para a demonstração não demorar.
    retry = (
        types.HttpRetryOptions(attempts=attempts, initial_delay=0.001, max_delay=0.002)
        if attempts
        else None
    )
    options = types.HttpOptions(httpx_client=sim.http_client, retry_options=retry)
    return genai.Client(api_key="falsa", http_options=options)


def generate_gemini(sim: Simulator, *, attempts: int | None = 3) -> str:
    client = gemini_client(sim, attempts=attempts)
    with gemini_errors():
        response = client.models.generate_content(model="modelo-gemini", contents=PROMPT)
    candidate = response.candidates[0] if response.candidates else None
    if candidate is None or candidate.content is None:
        reason = candidate.finish_reason if candidate else response.prompt_feedback
        raise RefusalError("gemini", f"resposta sem conteúdo: {reason}")
    return response.text or ""


def attempt(title: str, call: Callable[[Simulator], str], *replies: Reply) -> None:
    sim = simulate(*replies)
    try:
        outcome = f"ok: {call(sim)}"
    except LLMError as exc:
        outcome = f"{type(exc).__name__} (retryable={exc.retryable})"
    print(f"{title:<28} requisições={len(sim.requests)} {outcome}")


def main() -> None:
    ok = "Pedido enviado."

    print("429 duas vezes e depois sucesso")
    attempt("  openai (padrão)", generate_openai, error(429), error(429), openai_text(ok))
    attempt("  anthropic (padrão)", generate_anthropic, error(429), error(429), anthropic_text(ok))
    attempt("  gemini (attempts=3)", generate_gemini, error(429), error(429), gemini_text(ok))
    attempt("  gemini (padrão)", lambda sim: generate_gemini(sim, attempts=None), error(429))

    print("\n401: repetir não adianta")
    attempt("  openai", generate_openai, error(401))
    attempt("  anthropic", generate_anthropic, error(401))
    attempt("  gemini", generate_gemini, error(401))

    print("\nSem rede (o SDK também tenta de novo)")
    attempt("  openai", generate_openai, offline(), offline(), offline())
    attempt("  gemini (padrão)", lambda sim: generate_gemini(sim, attempts=None), offline())

    print("\nRecusa: status 200, mas sem resposta")
    attempt("  openai", generate_openai, openai_refusal())
    attempt("  anthropic", generate_anthropic, anthropic_refusal())
    attempt("  gemini", generate_gemini, gemini_refusal())


if __name__ == "__main__":
    main()
