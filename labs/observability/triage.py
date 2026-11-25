"""Cadeia LCEL: classifica a mensagem (saída estruturada) e roteia o atendimento."""

from typing import Any, Literal

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda, RunnablePassthrough
from pydantic import BaseModel, Field, ValidationError


class Triage(BaseModel):
    """Classificação de uma mensagem de cliente."""

    category: Literal["pedido", "financeiro", "outro"]
    urgent: bool = Field(description="Verdadeiro se exigir ação imediata")


PROMPT = ChatPromptTemplate.from_messages(
    [("system", "Classifique a mensagem do cliente."), ("human", "{message}")]
)


def build_triage_chain(model: BaseChatModel) -> Runnable[dict[str, Any], str]:
    # Classificação ilegível vira None; falhas de rede ou de chave continuam subindo.
    structured: Runnable[dict[str, Any], Any] = PROMPT | model.with_structured_output(Triage)
    unreadable: Runnable[dict[str, Any], Any] = RunnableLambda(lambda _: None)
    classify = structured.with_fallbacks(
        [unreadable], exceptions_to_handle=(OutputParserException, ValidationError)
    )
    route: Runnable[dict[str, Any], str] = RunnableBranch(
        # Na dúvida, uma pessoa atende.
        (
            lambda x: x["triage"] is None or x["triage"].urgent,
            RunnableLambda(lambda _: "fila-humana"),
        ),
        (lambda x: x["triage"].category == "pedido", RunnableLambda(lambda _: "agente-pedidos")),
        (lambda x: x["triage"].category == "financeiro", RunnableLambda(lambda _: "financeiro")),
        RunnableLambda(lambda _: "faq"),
    )
    chain: Runnable[dict[str, Any], str] = RunnablePassthrough.assign(triage=classify) | route
    return chain.with_config(run_name="triagem")
