"""Os três padrões com um LLM real: ``uv run pytest --live labs/langgraph_patterns``."""

import pytest

from labs.langgraph_patterns.chain_of_agents import ask
from labs.langgraph_patterns.supervisor import build_team, llm_worker
from labs.langgraph_patterns.tree_of_thoughts import solve
from labs.observability.tools import calcular_frete, consultar_pedido
from labs.shared.llm import live_chat_model

pytestmark = pytest.mark.live


def test_supervisor_team() -> None:
    model = live_chat_model()
    workers = [
        llm_worker(
            "pedidos",
            "consulta pedidos e calcula frete",
            model,
            "Use as ferramentas e responda só com fatos.",
            [consultar_pedido, calcular_frete],
        ),
        llm_worker("redator", "redige a resposta ao cliente", model, "Seja cordial e breve."),
    ]

    state = build_team(model, workers).invoke(
        {"task": "Informe ao cliente o status do pedido P-1002."}
    )

    assert state["status"] == "concluido"
    assert "pedidos" in [r["worker"] for r in state["reports"]]
    assert "pagamento" in state["answer"].lower()


def test_tree_of_thoughts() -> None:
    # Instância fácil (6 * 4 * 1 * 1) para não depender de um modelo grande.
    result = solve(live_chat_model(), [1, 1, 4, 6], breadth=3, proposals=5)

    assert result["solution"] is not None
    assert result["solution"].solved


def test_chain_of_agents() -> None:
    chunks = [
        "O projeto Aurora é liderado por Marina Duarte.",
        "A reunião de orçamento foi adiada para o próximo trimestre.",
        "Marina Duarte trabalha no escritório de Recife.",
    ]

    state = ask(live_chat_model(), "Em que cidade trabalha quem lidera o projeto Aurora?", chunks)

    assert "recife" in state["answer"].lower()
