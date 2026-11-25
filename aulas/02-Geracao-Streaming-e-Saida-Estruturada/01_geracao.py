"""Passo 1: a primeira chamada de geração em cada um dos três SDKs."""

import json
from dataclasses import dataclass

import anthropic
import openai
from google import genai
from google.genai import types
from simulador import Simulator, anthropic_text, gemini_text, openai_text, simulate

SYSTEM = "Responda em uma frase."
PROMPT = "Cadê o meu pedido?"


# --- Visto na aula 01 (versão reduzida) -------------------------------------------


@dataclass(frozen=True)
class ChatResult:
    text: str
    input_tokens: int
    output_tokens: int


# --- Novo neste passo -------------------------------------------------------------


def generate_openai(client: openai.OpenAI) -> ChatResult:
    response = client.responses.create(model="modelo-openai", instructions=SYSTEM, input=PROMPT)
    assert response.usage is not None
    return ChatResult(
        response.output_text, response.usage.input_tokens, response.usage.output_tokens
    )


def generate_anthropic(client: anthropic.Anthropic) -> ChatResult:
    message = client.messages.create(
        model="modelo-claude",
        max_tokens=1024,  # obrigatório na API do Claude
        system=SYSTEM,
        messages=[{"role": "user", "content": PROMPT}],
    )
    # A resposta é uma lista de blocos; só os de texto interessam aqui.
    text = "".join(block.text for block in message.content if block.type == "text")
    return ChatResult(text, message.usage.input_tokens, message.usage.output_tokens)


def generate_gemini(client: genai.Client) -> ChatResult:
    response = client.models.generate_content(
        model="modelo-gemini",
        contents=PROMPT,
        config=types.GenerateContentConfig(system_instruction=SYSTEM),
    )
    meta = response.usage_metadata
    assert meta is not None
    # Tokens de raciocínio vêm separados e são cobrados como saída.
    output = (meta.candidates_token_count or 0) + (meta.thoughts_token_count or 0)
    return ChatResult(response.text or "", meta.prompt_token_count or 0, output)


def show(title: str, sim: Simulator, result: ChatResult) -> None:
    path, body = sim.requests[0]
    print(f"== {title}")
    print(f"enviado : POST {path}")
    print(f"corpo   : {json.dumps(body, ensure_ascii=False)}")
    print(f"recebido: {result}\n")


def main() -> None:
    # A chave é falsa e nada sai da máquina: o simulador responde no lugar da API.
    sim = simulate(openai_text("Seu pedido foi enviado."))
    openai_client = openai.OpenAI(api_key="falsa", http_client=sim.http_client)
    show("OpenAI", sim, generate_openai(openai_client))

    sim = simulate(anthropic_text("Seu pedido foi enviado."))
    anthropic_client = anthropic.Anthropic(api_key="falsa", http_client=sim.http_client)
    show("Claude", sim, generate_anthropic(anthropic_client))

    sim = simulate(gemini_text("Seu pedido foi enviado."))
    options = types.HttpOptions(httpx_client=sim.http_client)
    gemini_client = genai.Client(api_key="falsa", http_options=options)
    show("Gemini", sim, generate_gemini(gemini_client))


if __name__ == "__main__":
    main()
