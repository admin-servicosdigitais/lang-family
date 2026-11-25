"""Tree-of-Thoughts: busca, poda e validação dos passos (offline)."""

from fractions import Fraction

import pytest
from langchain_core.messages import AIMessage, BaseMessage

from labs.langgraph_patterns.tests.game24 import fooled, game_model, oracle
from labs.langgraph_patterns.tree_of_thoughts import Node, Step, solve
from labs.shared.fake_model import ScriptedChatModel, tool_call


def replay(numbers: list[int], history: tuple[str, ...]) -> list[Fraction]:
    """Refaz a solução passo a passo para provar que ela é legítima."""
    node = Node(tuple(sorted(map(Fraction, numbers))))
    for line in history:
        a, op, b, *_ = line.split()
        next_node = node.apply(Step(a=a, op=op, b=b))
        assert next_node is not None, line
        node = next_node
    return list(node.numbers)


@pytest.mark.parametrize("numbers", [[4, 9, 10, 13], [3, 3, 8, 8], [1, 5, 5, 5]])
def test_finds_a_valid_solution(numbers: list[int]) -> None:
    result = solve(game_model(oracle), numbers, breadth=3, proposals=200)

    assert result["solution"] is not None
    assert len(result["solution"].history) == 3
    assert replay(numbers, result["solution"].history) == [24]


def test_unsolvable_instance_is_pruned_early() -> None:
    result = solve(game_model(oracle), [1, 1, 1, 1], breadth=3, proposals=200)

    assert result["solution"] is None
    assert result["frontier"] == []
    assert result["depth"] == 1


def test_wider_tree_recovers_from_a_misleading_evaluator() -> None:
    """O ponto do ToT: manter alternativas vivas quando a primeira escolha é ruim."""
    numbers = [2, 7, 8, 9]

    greedy = solve(game_model(fooled), numbers, breadth=1, proposals=200)
    tree = solve(game_model(fooled), numbers, breadth=3, proposals=200)

    assert greedy["solution"] is None
    assert tree["solution"] is not None
    assert replay(numbers, tree["solution"].history) == [24]
    assert tree["explored"] > greedy["explored"]


def test_frontier_is_expanded_in_parallel_branches() -> None:
    model = game_model(fooled)

    solve(model, [2, 7, 8, 9], breadth=3, proposals=200)

    proposals = [c for c in model.calls if "próximos passos" in c[0].text]
    # 1 nó na raiz + 3 no segundo nível + 3 no terceiro.
    assert len(proposals) == 7


INVALID_STEPS = [
    {"a": "99", "op": "+", "b": "1"},  # números que não existem
    {"a": "6", "op": "/", "b": "0"},  # divisão por zero
    {"a": "6", "op": "+", "b": "6"},  # só há um 6
    {"a": "abc", "op": "+", "b": "0"},  # não é número
]


@pytest.mark.parametrize("step", INVALID_STEPS)
def test_invalid_step_is_rejected(step: dict[str, str]) -> None:
    node = Node((Fraction(0), Fraction(4), Fraction(6)))

    assert node.apply(Step.model_validate(step)) is None


def test_hallucinated_steps_never_enter_the_tree() -> None:
    valid = {"a": "6", "op": "*", "b": "4"}

    def respond(_: list[BaseMessage], tools: list[str]) -> AIMessage:
        if tools == ["Proposals"]:
            return tool_call("Proposals", {"steps": [*INVALID_STEPS, valid]})
        return tool_call("Rating", {"score": 5})

    result = solve(ScriptedChatModel(responder=respond), [6, 4, 0], proposals=10)

    # Só o passo válido gerou um nó; no segundo nível nenhum passo se aplica.
    assert (result["explored"], result["depth"]) == (1, 2)
    assert result["solution"] is None


def test_model_cannot_claim_a_solution_it_did_not_compute() -> None:
    def respond(_: list[BaseMessage], tools: list[str]) -> AIMessage:
        if tools == ["Proposals"]:
            return tool_call("Proposals", {"steps": [{"a": "1", "op": "+", "b": "1"}]})
        return tool_call("Rating", {"score": 10})  # otimismo sem fundamento

    result = solve(ScriptedChatModel(responder=respond), [1, 1, 1, 1])

    assert result["solution"] is None
