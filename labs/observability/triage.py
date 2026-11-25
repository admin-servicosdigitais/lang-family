"""Cadeia LCEL: classifica a mensagem (saída estruturada) e roteia o atendimento."""

from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda, RunnablePassthrough
from pydantic import BaseModel, Field


class Triage(BaseModel):
    """Classificação de uma mensagem de cliente."""

    category: Literal["pedido", "financeiro", "outro"]
    urgent: bool = Field(description="Verdadeiro se exigir ação imediata")


PROMPT = ChatPromptTemplate.from_messages(
    [("system", "Classifique a mensagem do cliente."), ("human", "{message}")]
)


def build_triage_chain(model: BaseChatModel) -> Runnable[dict[str, Any], str]:
    classify = PROMPT | model.with_structured_output(Triage)
    route: Runnable[dict[str, Any], str] = RunnableBranch(
        (lambda x: x["triage"].urgent, RunnableLambda(lambda _: "fila-humana")),
        (lambda x: x["triage"].category == "pedido", RunnableLambda(lambda _: "agente-pedidos")),
        (lambda x: x["triage"].category == "financeiro", RunnableLambda(lambda _: "financeiro")),
        RunnableLambda(lambda _: "faq"),
    )
    chain: Runnable[dict[str, Any], str] = RunnablePassthrough.assign(triage=classify) | route
    return chain.with_config(run_name="triagem")
