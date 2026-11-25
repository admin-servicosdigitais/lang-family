"""Chain-of-Agents: informação espalhada pelo texto e contexto limitado (offline)."""

import re

import pytest
from langchain_core.messages import AIMessage, BaseMessage

from labs.langgraph_patterns.chain_of_agents import ask, split_text
from labs.shared.fake_model import ScriptedChatModel

QUESTION = "Em que cidade trabalha quem lidera o projeto Aurora?"
FILLER = "Relatório administrativo sem relação com o assunto. " * 20
CHUNKS = [
    f"{FILLER}O projeto Aurora é liderado por Marina. {FILLER}",
    FILLER,
    f"{FILLER}Marina trabalha no escritório de Recife. {FILLER}",
    FILLER,
]


def tag(name: str, text: str) -> str:
    match = re.search(rf"<{name}>(.*?)</{name}>", text, flags=re.DOTALL)
    return match[1] if match else ""


def detective() -> ScriptedChatModel:
    """Trabalhador/gerente por regras: só conecta os fatos se a CU os transportar."""

    def respond(messages: list[BaseMessage], _: list[str]) -> AIMessage:
        text = messages[-1].text
        notes = tag("anotacoes", text)
        if "<trecho>" not in text:  # gerente
            city = re.search(r"cidade=(\w+)", notes)
            return AIMessage(city[1] if city else "informação não encontrada")
        chunk = tag("trecho", text)
        if found := re.search(r"liderado por (\w+)", chunk):
            notes += f"lider={found[1]};"
        leader = re.search(r"lider=(\w+)", notes)
        if leader and (found := re.search(rf"{leader[1]} trabalha no escritório de (\w+)", chunk)):
            notes += f"cidade={found[1]};"
        return AIMessage(notes)

    return ScriptedChatModel(responder=respond)


def test_combines_facts_from_distant_chunks() -> None:
    state = ask(detective(), QUESTION, CHUNKS)

    assert state["answer"] == "Recife"
    assert state["history"] == [
        "lider=Marina;",
        "lider=Marina;",
        "lider=Marina;cidade=Recife;",
        "lider=Marina;cidade=Recife;",
    ]


def test_without_the_first_link_the_answer_is_not_found() -> None:
    state = ask(detective(), QUESTION, CHUNKS[1:])

    assert state["answer"] == "informação não encontrada"


def test_each_agent_sees_only_its_chunk_and_the_manager_sees_none() -> None:
    model = detective()

    ask(model, QUESTION, CHUNKS)

    prompts = [call[-1].text for call in model.calls]
    assert len(prompts) == len(CHUNKS) + 1
    for prompt, chunk in zip(prompts[:-1], CHUNKS, strict=True):
        assert tag("trecho", prompt) == chunk
        assert len(prompt) < len(chunk) + 300
    assert "<trecho>" not in prompts[-1]
    assert "Relatório administrativo" not in prompts[-1]


def test_communication_unit_is_bounded() -> None:
    verbose = ScriptedChatModel(responder=lambda *_: AIMessage("x" * 5000))

    state = ask(verbose, QUESTION, CHUNKS, notes_limit=100)

    assert [len(notes) for notes in state["history"]] == [100] * len(CHUNKS)


@pytest.mark.parametrize("count", [0, 60])
def test_handles_any_number_of_chunks(count: int) -> None:
    model = detective()

    ask(model, QUESTION, [FILLER] * count)

    assert len(model.calls) == count + 1


def test_split_text_respects_chunk_size() -> None:
    text = "\n\n".join(CHUNKS)

    chunks = split_text(text, chunk_size=500, chunk_overlap=50)

    assert all(len(c) <= 500 for c in chunks)
    # Dividir não pode perder nem cortar os fatos relevantes.
    joined = " ".join(chunks)
    assert "liderado por Marina" in joined
    assert "escritório de Recife" in joined
