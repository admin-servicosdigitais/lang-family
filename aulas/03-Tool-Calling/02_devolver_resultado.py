"""Passo 2: uma volta completa, com o resultado devolvido ao modelo."""

import json
from typing import Any

import anthropic
import openai
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from simulador import (
    Simulator,
    anthropic_text,
    anthropic_tool_calls,
    gemini_text,
    gemini_tool_calls,
    openai_text,
    openai_tool_calls,
    simulate,
)

PROMPT = "Cadê o pedido P-1001?"
NAME = "consultar_pedido"
DESCRIPTION = "Consulta o status de um pedido."


# --- Visto no passo 1 -------------------------------------------------------------


class OrderArgs(BaseModel):
    order_id: str = Field(description="Número do pedido, ex.: P-1001")


SCHEMA = OrderArgs.model_json_schema()


# --- Novo neste passo: executar e responder ---------------------------------------


def order_status(order_id: str) -> str:
    return {"P-1001": "enviado", "P-1002": "aguardando pagamento"}[order_id]


def round_openai(sim: Simulator) -> str:
    client = openai.OpenAI(api_key="falsa", http_client=sim.http_client)
    tools = [{"type": "function", "name": NAME, "parameters": SCHEMA, "strict": False}]
    request: dict[str, Any] = {"model": "modelo-openai", "input": PROMPT, "tools": tools}

    first = client.responses.create(**request)
    outputs = []
    for item in first.output:
        if item.type == "function_call":
            output = order_status(**json.loads(item.arguments))
            outputs.append(
                {"type": "function_call_output", "call_id": item.call_id, "output": output}
            )

    # A API guarda a conversa: vai só o resultado e o id da resposta anterior.
    request = request | {"input": outputs, "previous_response_id": first.id}
    return str(client.responses.create(**request).output_text)


def round_anthropic(sim: Simulator) -> str:
    client = anthropic.Anthropic(api_key="falsa", http_client=sim.http_client)
    tools: Any = [{"name": NAME, "description": DESCRIPTION, "input_schema": SCHEMA}]
    messages: list[Any] = [{"role": "user", "content": PROMPT}]

    first = client.messages.create(
        model="modelo-claude", max_tokens=1024, messages=messages, tools=tools
    )
    results = []
    for block in first.content:
        if block.type == "tool_use":
            arguments: dict[str, Any] = dict(block.input)
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": order_status(**arguments),
                }
            )

    # A API não guarda nada: a conversa inteira é reenviada.
    messages = [
        *messages,
        {"role": "assistant", "content": first.content},  # blocos intactos
        {"role": "user", "content": results},
    ]
    second = client.messages.create(
        model="modelo-claude", max_tokens=1024, messages=messages, tools=tools
    )
    return "".join(block.text for block in second.content if block.type == "text")


def round_gemini(sim: Simulator) -> str:
    options = types.HttpOptions(httpx_client=sim.http_client)
    client = genai.Client(api_key="falsa", http_options=options)
    declaration = types.FunctionDeclaration(
        name=NAME, description=DESCRIPTION, parameters_json_schema=SCHEMA
    )
    config = types.GenerateContentConfig(
        tools=[types.Tool(function_declarations=[declaration])],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    contents: list[Any] = [types.Content(role="user", parts=[types.Part(text=PROMPT)])]

    first = client.models.generate_content(model="modelo-gemini", contents=contents, config=config)
    assert first.candidates
    assert first.candidates[0].content
    parts = first.candidates[0].content.parts or []
    replies = []
    for part in parts:
        call = part.function_call
        if call is None:
            continue
        output = order_status(**dict(call.args or {}))
        reply = types.Part.from_function_response(name=NAME, response={"result": output})
        assert reply.function_response is not None
        reply.function_response.id = call.id
        replies.append(reply)

    # Também reenvia tudo; as partes do modelo voltam com a thought signature.
    contents += [
        types.Content(role="model", parts=parts),
        types.Content(role="user", parts=replies),
    ]
    second = client.models.generate_content(model="modelo-gemini", contents=contents, config=config)
    return second.text or ""


def main() -> None:
    call = (NAME, {"order_id": "P-1001"})
    answer = "Seu pedido foi enviado."
    for title, round_trip, sim, history in (
        (
            "OpenAI",
            round_openai,
            simulate(openai_tool_calls(call), openai_text(answer)),
            ("previous_response_id", "input"),
        ),
        (
            "Claude",
            round_anthropic,
            simulate(anthropic_tool_calls(call), anthropic_text(answer)),
            ("messages",),
        ),
        (
            "Gemini",
            round_gemini,
            simulate(gemini_tool_calls(call), gemini_text(answer)),
            ("contents",),
        ),
    ):
        text = round_trip(sim)
        second = {key: sim.requests[1][key] for key in history}
        print(f"== {title}: {len(sim.requests)} requisições")
        print("histórico na 2ª requisição:")
        print(json.dumps(second, ensure_ascii=False, indent=2))
        print(f"resposta final: {text}\n")


if __name__ == "__main__":
    main()
