"""Passo 5: o Chain-of-Agents do laboratório (``langgraph_patterns/chain_of_agents.py``).

O modelo por regras vem dos testes do laboratório. Ele e o ``main`` são só para a aula.
"""

import operator
import re
from collections.abc import Sequence
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from modelo_falso import ScriptedChatModel

# --- Do laboratório: langgraph_patterns/chain_of_agents.py --------------------------


WORKER_PROMPT = (
    "Você é um agente trabalhador em uma cadeia. Atualize as anotações com o que o trecho "
    "acrescenta para responder à pergunta. Preserve as anotações anteriores que sejam "
    "úteis. Devolva somente as anotações, em até {limit} caracteres."
)


MANAGER_PROMPT = (
    "Você é o agente gerente. Responda à pergunta usando somente as anotações. "
    "Se elas não bastarem, diga que a informação não foi encontrada."
)


class ChainState(TypedDict, total=False):
    question: str
    chunks: list[str]
    index: int
    notes: str
    history: Annotated[list[str], operator.add]
    answer: str


def split_text(text: str, chunk_size: int = 2000, chunk_overlap: int = 100) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


def build_chain(
    worker: BaseChatModel, manager: BaseChatModel | None = None, *, notes_limit: int = 1000
) -> CompiledStateGraph[Any, Any, Any, Any]:
    if notes_limit < 1:
        raise ValueError("notes_limit precisa ser positivo")
    manager = manager or worker

    def work(state: ChainState) -> ChainState:
        index = state.get("index", 0)
        reply = worker.invoke(
            [
                SystemMessage(WORKER_PROMPT.format(limit=notes_limit)),
                HumanMessage(
                    f"<pergunta>{state['question']}</pergunta>\n"
                    f"<anotacoes>{state.get('notes', '')}</anotacoes>\n"
                    f"<trecho>{state['chunks'][index]}</trecho>"
                ),
            ]
        )
        # O limite é imposto em código: a CU nunca cresce junto com o documento.
        # Resposta vazia (recusa, corte) não pode apagar o que já foi coletado.
        notes = str(reply.text).strip()[:notes_limit] or state.get("notes", "")
        return {"notes": notes, "history": [notes], "index": index + 1}

    def manage(state: ChainState) -> ChainState:
        reply = manager.invoke(
            [
                SystemMessage(MANAGER_PROMPT),
                HumanMessage(
                    f"<pergunta>{state['question']}</pergunta>\n"
                    f"<anotacoes>{state.get('notes', '')}</anotacoes>"
                ),
            ]
        )
        return {"answer": str(reply.text)}

    def next_step(state: ChainState) -> str:
        return "worker" if state.get("index", 0) < len(state["chunks"]) else "manager"

    graph = StateGraph(ChainState)
    graph.add_node("worker", work)
    graph.add_node("manager", manage)
    graph.add_conditional_edges(START, next_step, ["worker", "manager"])
    graph.add_conditional_edges("worker", next_step, ["worker", "manager"])
    graph.add_edge("manager", END)
    return graph.compile(name="chain_of_agents")


def ask(
    worker: BaseChatModel,
    question: str,
    chunks: Sequence[str],
    *,
    notes_limit: int = 1000,
) -> ChainState:
    app = build_chain(worker, notes_limit=notes_limit)
    # Cada trecho é um passo do grafo: o limite de recursão acompanha a quantidade.
    config: Any = {"recursion_limit": len(chunks) + 10}
    result: Any = app.invoke({"question": question, "chunks": list(chunks)}, config)
    state: ChainState = result
    return state


# --- Dos testes do laboratório ------------------------------------------------------

QUESTION = "Em que cidade trabalha quem lidera o projeto Aurora?"
FILLER = "Relatório administrativo sem relação com o assunto. " * 10
CHUNKS = [
    f"{FILLER}O projeto Aurora é liderado por Marina. {FILLER}",
    FILLER,
    f"{FILLER}Marina trabalha no escritório de Recife. {FILLER}",
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


# --- Somente da aula ----------------------------------------------------------------


def main() -> None:
    chunks = split_text("\n\n".join(CHUNKS), chunk_size=600, chunk_overlap=50)
    model = detective()
    state = ask(model, QUESTION, chunks)
    print(f"{len(chunks)} trechos, {len(model.calls)} chamadas ao modelo")
    for number, notes in enumerate(state["history"], start=1):
        print(f"  trabalhador {number}: {notes!r}")
    print(f"resposta: {state['answer']}")
    short = ask(detective(), QUESTION, chunks, notes_limit=14)
    print(f"com notes_limit=14: notas={short['notes']!r} resposta={short['answer']}")


if __name__ == "__main__":
    main()
