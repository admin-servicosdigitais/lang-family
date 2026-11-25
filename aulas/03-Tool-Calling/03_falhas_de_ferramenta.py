"""Passo 3: falhas da ferramenta voltam ao modelo como dado, para ele se corrigir."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import anthropic
from google.genai import types
from pydantic import BaseModel, ValidationError
from simulador import Simulator, anthropic_text, anthropic_tool_calls, simulate

# --- Visto na aula 01, passo 5 ----------------------------------------------------


@dataclass(frozen=True)
class ToolRun:
    name: str
    arguments: dict[str, Any]
    output: str
    is_error: bool = False


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    function: Callable[..., object]

    @property
    def json_schema(self) -> dict[str, Any]:
        return self.args_model.model_json_schema()

    def run(self, arguments: dict[str, Any]) -> ToolRun:
        try:
            args = self.args_model.model_validate(arguments)
            # dict(args) preserva modelos aninhados; model_dump() os viraria em dicts.
            output = str(self.function(**dict(args)))
        except ValidationError as error:
            return ToolRun(self.name, arguments, f"argumentos inválidos: {error}", True)
        except Exception as error:
            return ToolRun(self.name, arguments, f"{type(error).__name__}: {error}", True)
        return ToolRun(self.name, arguments, output)


def run_tool(tools: list[Tool], name: str, arguments: dict[str, Any]) -> ToolRun:
    for tool in tools:
        if tool.name == name:
            return tool.run(arguments)
    return ToolRun(name, arguments, f"ferramenta desconhecida: {name}", True)


# --- Novo neste passo: a OpenAI manda texto, que pode não ser JSON ----------------


def run_raw_call(tools: list[Tool], name: str, raw_arguments: str) -> ToolRun:
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as error:
        return ToolRun(name, {}, f"argumentos inválidos: {error}", True)
    return run_tool(tools, name, arguments)


# --- Novo neste passo: como cada API é avisada da falha ---------------------------


def to_openai(call_id: str, run: ToolRun) -> dict[str, Any]:
    # Não há campo de erro: a mensagem vai no próprio output.
    return {"type": "function_call_output", "call_id": call_id, "output": run.output}


def to_anthropic(call_id: str, run: ToolRun) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": call_id,
        "content": run.output,
        "is_error": run.is_error,
    }


def to_gemini(call_id: str, run: ToolRun) -> types.Part:
    # A chave "error" ou "result" é o que distingue falha de sucesso.
    key = "error" if run.is_error else "result"
    reply = types.Part.from_function_response(name=run.name, response={key: run.output})
    assert reply.function_response is not None
    reply.function_response.id = call_id
    return reply


# --- Novo neste passo: o modelo se corrige ----------------------------------------


class OrderArgs(BaseModel):
    order_id: str


def order_status(order_id: str) -> str:
    return {"P-1001": "enviado", "P-1002": "aguardando pagamento"}[order_id]


ORDER_TOOL = Tool("consultar_pedido", "Consulta um pedido.", OrderArgs, order_status)


def self_correction(sim: Simulator) -> str:
    client = anthropic.Anthropic(api_key="falsa", http_client=sim.http_client)
    tools: Any = [{"name": ORDER_TOOL.name, "input_schema": ORDER_TOOL.json_schema}]
    messages: list[Any] = [{"role": "user", "content": "Cadê o pedido P-1001?"}]

    def ask() -> anthropic.types.Message:
        return client.messages.create(
            model="modelo-claude", max_tokens=1024, messages=messages, tools=tools
        )

    def answer(message: anthropic.types.Message) -> None:
        results = []
        for block in message.content:
            if block.type == "tool_use":
                run = run_tool([ORDER_TOOL], block.name, dict(block.input))
                print(f"  pediu {dict(block.input)} -> is_error={run.is_error}")
                results.append(to_anthropic(block.id, run))
        messages.append({"role": "assistant", "content": message.content})
        messages.append({"role": "user", "content": results})

    # Duas voltas escritas à mão. Repare na repetição: o passo 4 a transforma em laço.
    answer(ask())  # 1ª tentativa: argumento com nome errado
    answer(ask())  # 2ª tentativa: o modelo leu o erro e corrigiu
    final = ask()
    return "".join(block.text for block in final.content if block.type == "text")


def main() -> None:
    print("1. Nenhum caso lança exceção")
    cases: list[tuple[str, str, dict[str, Any]]] = [
        ("válido", "consultar_pedido", {"order_id": "P-1001"}),
        ("campo ausente", "consultar_pedido", {}),
        ("tipo errado", "consultar_pedido", {"order_id": 1001}),
        ("função falha", "consultar_pedido", {"order_id": "P-9999"}),
        ("não existe", "apagar_tudo", {}),
        ("argumento extra", "consultar_pedido", {"order_id": "P-1001", "apagar": True}),
    ]
    for label, name, arguments in cases:
        run = run_tool([ORDER_TOOL], name, arguments)
        print(f"  {label:<16} is_error={run.is_error!s:<5} {run.output.splitlines()[0]}")
    cut = run_raw_call([ORDER_TOOL], "consultar_pedido", '{"order_id": "P-10')
    print(f"  {'JSON cortado':<16} is_error={cut.is_error!s:<5} {cut.output}")

    print("\n2. A mesma falha no formato de cada API")
    failed = run_tool([ORDER_TOOL], "consultar_pedido", {"order_id": "P-9999"})
    print("  openai   :", to_openai("call_1", failed))
    print("  anthropic:", to_anthropic("toolu_1", failed))
    part = to_gemini("call_1", failed)
    print("  gemini   :", part.model_dump(exclude_none=True))

    print("\n3. O modelo se corrige")
    sim = simulate(
        anthropic_tool_calls(("consultar_pedido", {"pedido": "P-1001"})),
        anthropic_tool_calls(("consultar_pedido", {"order_id": "P-1001"})),
        anthropic_text("Seu pedido foi enviado."),
    )
    print(f"  resposta final: {self_correction(sim)}")


if __name__ == "__main__":
    main()
