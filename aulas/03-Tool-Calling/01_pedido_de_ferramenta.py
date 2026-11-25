"""Passo 1: o modelo não executa nada, ele pede que a aplicação execute."""

import json
from typing import Any

import anthropic
import openai
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from simulador import (
    Simulator,
    anthropic_tool_calls,
    gemini_tool_calls,
    openai_tool_calls,
    simulate,
)

PROMPT = "Cadê o pedido P-1001?"
NAME = "consultar_pedido"
DESCRIPTION = "Consulta o status de um pedido."


class OrderArgs(BaseModel):
    order_id: str = Field(description="Número do pedido, ex.: P-1001")


# O mesmo JSON Schema vai para as três APIs; o que muda é o envelope.
SCHEMA = OrderArgs.model_json_schema()


def ask_openai(sim: Simulator) -> None:
    client = openai.OpenAI(api_key="falsa", http_client=sim.http_client)
    tools: Any = [
        {
            "type": "function",
            "name": NAME,
            "description": DESCRIPTION,
            "parameters": SCHEMA,
            "strict": False,
        }
    ]
    response = client.responses.create(model="modelo-openai", input=PROMPT, tools=tools)
    for item in response.output:
        if item.type == "function_call":
            print(f"  id       : {item.call_id}")
            print(f"  nome     : {item.name}")
            print(f"  recebido : {item.arguments!r} ({type(item.arguments).__name__})")
            print(f"  convertido: {json.loads(item.arguments)}")


def ask_anthropic(sim: Simulator) -> None:
    client = anthropic.Anthropic(api_key="falsa", http_client=sim.http_client)
    tools: Any = [{"name": NAME, "description": DESCRIPTION, "input_schema": SCHEMA}]
    message = client.messages.create(
        model="modelo-claude",
        max_tokens=1024,
        messages=[{"role": "user", "content": PROMPT}],
        tools=tools,
    )
    print(f"  motivo   : stop_reason={message.stop_reason}")
    print(f"  blocos   : {[block.type for block in message.content]}")
    for block in message.content:
        if block.type == "tool_use":
            print(f"  id       : {block.id}")
            print(f"  nome     : {block.name}")
            print(f"  recebido : {block.input!r} ({type(block.input).__name__})")


def ask_gemini(sim: Simulator) -> None:
    options = types.HttpOptions(httpx_client=sim.http_client)
    client = genai.Client(api_key="falsa", http_options=options)
    declaration = types.FunctionDeclaration(
        name=NAME, description=DESCRIPTION, parameters_json_schema=SCHEMA
    )
    config = types.GenerateContentConfig(
        tools=[types.Tool(function_declarations=[declaration])],
        # Sem isto, o SDK poderia executar funções Python sozinho.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    response = client.models.generate_content(model="modelo-gemini", contents=PROMPT, config=config)
    for call in response.function_calls or []:
        print(f"  id       : {call.id}")
        print(f"  nome     : {call.name}")
        print(f"  recebido : {call.args!r} ({type(call.args).__name__})")


def main() -> None:
    call = (NAME, {"order_id": "P-1001"})
    for title, ask, reply in (
        ("OpenAI", ask_openai, openai_tool_calls(call)),
        ("Claude", ask_anthropic, anthropic_tool_calls(call)),
        ("Gemini", ask_gemini, gemini_tool_calls(call)),
    ):
        sim = simulate(reply)
        print(f"== {title}")
        ask(sim)
        print(f"  declaração enviada: {json.dumps(sim.requests[0]['tools'], ensure_ascii=False)}\n")
    print("Nenhuma função foi executada: até aqui só existe o pedido.")


if __name__ == "__main__":
    main()
