"""Passo 5: o Tree-of-Thoughts do laboratório (``langgraph_patterns/tree_of_thoughts.py``).

O modelo roteirizado vem dos testes do laboratório. Ele e o ``main`` são só para a aula.
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from itertools import permutations
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send
from modelo_falso import ScriptedChatModel, tool_call
from pydantic import BaseModel, Field, ValidationError

# --- Do laboratório: langgraph_patterns/tree_of_thoughts.py -------------------------


TARGET = Fraction(24)


# Números legítimos do jogo são curtos; notação como "1e999999999" custaria caro.
MAX_NUMBER_LENGTH = 20


PROPOSE_PROMPT = (
    "Jogo do 24. Proponha até {k} próximos passos. Cada passo combina DOIS dos números "
    "disponíveis com uma operação; o resultado substitui os dois."
)


EVALUATE_PROMPT = (
    "Jogo do 24. Dê uma nota de 0 a 10 para a chance de os números restantes ainda "
    "chegarem a exatamente 24."
)


class Step(BaseModel):
    a: str = Field(description="Primeiro número, ex.: '8' ou '8/3'")
    op: Literal["+", "-", "*", "/"]
    b: str = Field(description="Segundo número")


class Proposals(BaseModel):
    """Próximos passos candidatos."""

    steps: list[Step]


class Rating(BaseModel):
    """Avaliação de um estado."""

    score: float = Field(ge=0, le=10)


@dataclass(frozen=True)
class Node:
    numbers: tuple[Fraction, ...]
    history: tuple[str, ...] = ()
    score: float = 0.0

    @property
    def solved(self) -> bool:
        return self.numbers == (TARGET,)

    def apply(self, step: Step) -> "Node | None":
        """Aplica o passo se for válido; passos inventados pelo modelo são descartados."""
        if any(len(t) > MAX_NUMBER_LENGTH or "e" in t.lower() for t in (step.a, step.b)):
            return None
        try:
            a, b = Fraction(step.a), Fraction(step.b)
            remaining = list(self.numbers)
            remaining.remove(a)
            remaining.remove(b)
            result = {
                "+": lambda: a + b,
                "-": lambda: a - b,
                "*": lambda: a * b,
                "/": lambda: a / b,
            }[step.op]()
        except (ValueError, ZeroDivisionError):
            return None
        line = f"{a} {step.op} {b} = {result}"
        return Node((*sorted(remaining), result), (*self.history, line))


def describe(numbers: Sequence[Fraction]) -> str:
    return "Números disponíveis: " + ", ".join(str(n) for n in numbers)


def parse_numbers(text: str) -> list[Fraction]:
    match = re.search(r"Números disponíveis: (.+)", text)
    assert match, text
    return [Fraction(part.strip()) for part in match[1].split(",")]


def _collect(current: list[Node], update: list[Node] | None) -> list[Node]:
    """Redutor: acumula os ramos paralelos; ``None`` esvazia para o próximo nível."""
    return [] if update is None else current + update


class SearchState(TypedDict, total=False):
    frontier: list[Node]
    candidates: Annotated[list[Node], _collect]
    solution: Node | None
    depth: int
    explored: int


class _Branch(TypedDict):
    node: Node


def _structured(model: BaseChatModel, schema: type[BaseModel]) -> Runnable[Any, Any]:
    """Resposta fora do formato vira None; falhas de rede ou de chave continuam subindo."""
    structured: Runnable[Any, Any] = model.with_structured_output(schema)
    unreadable: Runnable[Any, Any] = RunnableLambda(lambda _: None)
    return structured.with_fallbacks(
        [unreadable], exceptions_to_handle=(OutputParserException, ValidationError)
    )


def build_tot(
    model: BaseChatModel, *, breadth: int = 5, proposals: int = 8
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """``breadth`` é quantos estados sobrevivem a cada nível (1 = busca gulosa)."""
    proposer = _structured(model, Proposals)
    rater = _structured(model, Rating)

    def fan_out(state: SearchState) -> list[Send] | str:
        if state.get("solution") or not state.get("frontier"):
            return END
        return [Send("propose", _Branch(node=node)) for node in state["frontier"]]

    def propose(state: _Branch) -> SearchState:
        node = state["node"]
        reply: Any = proposer.invoke(
            [
                SystemMessage(PROPOSE_PROMPT.format(k=proposals)),
                HumanMessage(describe(node.numbers)),
            ]
        )
        if reply is None:
            return {"candidates": []}
        children = (node.apply(step) for step in reply.steps[:proposals])
        return {"candidates": [child for child in children if child is not None]}

    def evaluate(state: SearchState) -> SearchState:
        # Estados com os mesmos números são equivalentes: avalia-se um só.
        unique = {tuple(sorted(n.numbers)): n for n in state.get("candidates", [])}
        nodes = list(unique.values())
        solution = next((n for n in nodes if n.solved), None)
        survivors: list[Node] = []
        open_nodes = [n for n in nodes if len(n.numbers) > 1]
        if solution is None and open_nodes:
            ratings: list[Any] = rater.batch(
                [
                    [SystemMessage(EVALUATE_PROMPT), HumanMessage(describe(n.numbers))]
                    for n in open_nodes
                ]
            )
            scored = [
                Node(n.numbers, n.history, r.score)
                for n, r in zip(open_nodes, ratings, strict=True)
                # Nota zero = ramo impossível; sem nota legível, o ramo também sai.
                if r is not None and r.score > 0
            ]
            survivors = sorted(scored, key=lambda n: n.score, reverse=True)[:breadth]
        return {
            "frontier": survivors,
            "candidates": None,  # type: ignore[typeddict-item]
            "solution": solution,
            "depth": state.get("depth", 0) + 1,
            "explored": state.get("explored", 0) + len(nodes),
        }

    graph = StateGraph(SearchState)
    graph.add_node("propose", propose, input_schema=_Branch)
    graph.add_node("evaluate", evaluate)
    graph.add_conditional_edges(START, fan_out, ["propose", END])
    graph.add_edge("propose", "evaluate")
    graph.add_conditional_edges("evaluate", fan_out, ["propose", END])
    return graph.compile(name="tree_of_thoughts")


def solve(model: BaseChatModel, numbers: Sequence[int], **options: int) -> SearchState:
    root = Node(tuple(sorted(Fraction(n) for n in numbers)))
    result = build_tot(model, **options).invoke({"frontier": [root]})
    final: SearchState = result  # type: ignore[assignment]
    return final


# --- Dos testes do laboratório: game24.py -------------------------------------------


Rate = Callable[[list[Fraction]], float]


def all_steps(numbers: list[Fraction]) -> list[Step]:
    return [Step(a=str(a), op=op, b=str(b)) for a, b in permutations(numbers, 2) for op in "+-*/"]


@cache
def reachable(numbers: tuple[Fraction, ...]) -> bool:
    if len(numbers) == 1:
        return numbers[0] == TARGET
    node = Node(numbers)
    children = (node.apply(step) for step in all_steps(list(numbers)))
    return any(reachable(c.numbers) for c in children if c is not None)


def oracle(numbers: list[Fraction]) -> float:
    """Avaliador perfeito: 10 se 24 ainda é alcançável, 0 caso contrário."""
    return 10.0 if reachable(tuple(sorted(numbers))) else 0.0


def fooled(numbers: list[Fraction]) -> float:
    """Avaliador imperfeito: tem falsos positivos em estados com um número perto de 24.

    Estados viáveis recebem 6, mas "armadilhas" sem solução recebem 8.
    """
    if oracle(numbers):
        return 6.0
    return 8.0 if any(abs(n - TARGET) <= 2 for n in numbers) else 1.0


def game_model(rate: Rate) -> ScriptedChatModel:
    def respond(messages: list[BaseMessage], tools: list[str]) -> AIMessage:
        numbers = parse_numbers(messages[-1].text)
        if tools == ["Proposals"]:
            return tool_call("Proposals", {"steps": [s.model_dump() for s in all_steps(numbers)]})
        return tool_call("Rating", {"score": rate(numbers)})

    return ScriptedChatModel(responder=respond)


# --- Somente da aula ----------------------------------------------------------------


def main() -> None:
    for numbers, rate, label in (
        ([4, 9, 10, 13], oracle, "avaliador perfeito"),
        ([2, 7, 8, 9], fooled, "avaliador imperfeito"),
    ):
        for breadth in (1, 3):
            model = game_model(rate)
            result = solve(model, numbers, breadth=breadth, proposals=200)
            solution = result["solution"]
            print(f"{numbers} {label}, largura {breadth}:")
            print(f"  chamadas={len(model.calls)} estados={result['explored']}")
            print(f"  solução={list(solution.history) if solution else None}")


if __name__ == "__main__":
    main()
