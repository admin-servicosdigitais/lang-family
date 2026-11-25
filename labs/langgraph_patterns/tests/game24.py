"""Modelos roteirizados para o Jogo do 24: um propositor e dois avaliadores."""

from collections.abc import Callable
from fractions import Fraction
from functools import cache
from itertools import permutations

from langchain_core.messages import AIMessage, BaseMessage

from labs.langgraph_patterns.tree_of_thoughts import TARGET, Node, Step, parse_numbers
from labs.shared.fake_model import ScriptedChatModel, tool_call

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
