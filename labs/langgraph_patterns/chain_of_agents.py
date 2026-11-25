"""Chain-of-Agents (Zhang et al., 2024): contexto longo lido por uma cadeia de agentes.

    trabalhador(trecho 1) -> trabalhador(trecho 2) -> ... -> gerente

Cada trabalhador vê apenas o seu trecho e a "unidade de comunicação" (CU) recebida
do anterior. O gerente responde só com a CU final, sem nunca ver o texto inteiro.
"""

import operator
from collections.abc import Sequence
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

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
