"""Passo 4: o laço de ferramentas, com chamadas paralelas e limite de turnos.

Usa só o Claude, para o laço ficar em primeiro plano. O passo 5 traz os três.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import anthropic
from pydantic import BaseModel, ValidationError
from simulador import Simulator, anthropic_text, anthropic_tool_calls, simulate

# --- Visto nas aulas 01 e passo 3 -------------------------------------------------


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
class ToolRun:
    name: str
    arguments: dict[str, Any]
    output: str
    is_error: bool = False


@dataclass(frozen=True)
class ChatResult:
    text: str
    usage: Usage
    tool_runs: tuple[ToolRun, ...]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    function: Callable[..., object]

    def run(self, arguments: dict[str, Any]) -> ToolRun:
        try:
            args = self.args_model.model_validate(arguments)
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


# --- Novo neste passo: o laço -----------------------------------------------------


class ToolLoopError(Exception):
    pass


def run_tools(sim: Simulator, prompt: str, tools: list[Tool], *, max_turns: int = 5) -> ChatResult:
    client = anthropic.Anthropic(api_key="falsa", http_client=sim.http_client)
    definitions: Any = [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.args_model.model_json_schema(),
        }
        for t in tools
    ]
    messages: list[Any] = [{"role": "user", "content": prompt}]
    usage = Usage()
    runs: list[ToolRun] = []
    for turn in range(1, max_turns + 1):
        message = client.messages.create(
            model="modelo-claude", max_tokens=1024, messages=messages, tools=definitions
        )
        usage += Usage(message.usage.input_tokens, message.usage.output_tokens)
        if message.stop_reason != "tool_use":
            print(f"  turno {turn}: resposta final")
            text = "".join(block.text for block in message.content if block.type == "text")
            return ChatResult(text, usage, tuple(runs))
        results = []
        for block in message.content:  # pode haver vários pedidos no mesmo turno
            if block.type != "tool_use":
                continue
            run = run_tool(tools, block.name, dict(block.input))
            runs.append(run)
            print(f"  turno {turn}: {run.name} {run.arguments} -> {run.output}")
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": run.output,
                    "is_error": run.is_error,
                }
            )
        # Todos os resultados do turno vão em uma única mensagem de usuário.
        messages = [
            *messages,
            {"role": "assistant", "content": message.content},
            {"role": "user", "content": results},
        ]
    raise ToolLoopError(f"sem resposta final após {max_turns} turnos")


class OrderArgs(BaseModel):
    order_id: str


class FreightArgs(BaseModel):
    total: float
    regiao: str


def script() -> Simulator:
    return simulate(
        anthropic_tool_calls(
            ("consultar_pedido", {"order_id": "P-1001"}),
            ("consultar_pedido", {"order_id": "P-1002"}),
        ),
        anthropic_tool_calls(("calcular_frete", {"total": 350, "regiao": "sudeste"})),
        anthropic_text("O P-1001 tem frete grátis; o P-1002 aguarda pagamento."),
    )


def main() -> None:
    orders = {
        "P-1001": "enviado; total 350; sudeste",
        "P-1002": "aguardando pagamento; total 120; norte",
    }
    executed: list[str] = []

    def order_status(order_id: str) -> str:
        executed.append("consultar_pedido")
        return orders[order_id]

    def freight(total: float, regiao: str) -> str:
        executed.append("calcular_frete")
        return "grátis" if total > 299 and regiao == "sudeste" else "R$ 29,90"

    tools = [
        Tool("consultar_pedido", "Consulta um pedido.", OrderArgs, order_status),
        Tool("calcular_frete", "Calcula o frete.", FreightArgs, freight),
    ]
    prompt = "Como estão o P-1001 e o P-1002, e qual o frete do primeiro?"

    print("Com limite suficiente")
    sim = script()
    result = run_tools(sim, prompt, tools)
    print(f"  texto     : {result.text}")
    print(f"  requisições={len(sim.requests)} execuções={len(result.tool_runs)} {result.usage}")

    print("\nCom max_turns=2")
    executed.clear()
    sim = script()
    try:
        run_tools(sim, prompt, tools, max_turns=2)
    except ToolLoopError as error:
        print(f"  ToolLoopError: {error}")
    print(f"  requisições={len(sim.requests)} execuções={len(executed)}")
    print("  o resultado da última execução nunca chegou ao modelo")


if __name__ == "__main__":
    main()
