"""Passo 5: ferramentas e triagem do laboratório.

Reproduz ``labs/observability/tools.py`` e ``triage.py``. O ``main`` é só para a aula.
"""

from typing import Any, Literal

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.tools import ToolException, tool
from modelo_falso import ScriptedChatModel, tool_call
from pydantic import BaseModel, Field, ValidationError

# --- Do laboratório: observability/tools.py -----------------------------------------


ORDERS = {
    "P-1001": {"status": "enviado", "total": 350.0, "regiao": "sudeste"},
    "P-1002": {"status": "aguardando pagamento", "total": 120.0, "regiao": "norte"},
}


@tool
def consultar_pedido(order_id: str) -> str:
    """Consulta status, valor e região de um pedido, pelo número (ex.: P-1001)."""
    order = ORDERS.get(order_id.strip().upper())
    if order is None:
        raise ToolException(f"pedido {order_id} não encontrado")
    return f"status: {order['status']}; total: R$ {order['total']:.2f}; região: {order['regiao']}"


@tool
def calcular_frete(total: float, regiao: str) -> str:
    """Calcula o frete a partir do valor do pedido e da região do país."""
    if total > 299 and regiao.strip().lower() in {"sul", "sudeste"}:
        return "frete grátis"
    return "frete: R$ 29,90"


# Erros previstos voltam ao modelo como mensagem, sem derrubar o agente.
consultar_pedido.handle_tool_error = True

TOOLS = [consultar_pedido, calcular_frete]


# --- Do laboratório: observability/triage.py ----------------------------------------


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


# --- Somente da aula ----------------------------------------------------------------


def main() -> None:
    print([t.name for t in TOOLS])
    print(consultar_pedido.invoke({"order_id": " p-1001 "}))
    replies = {
        "pedido": tool_call("Triage", {"category": "pedido", "urgent": False}),
        "urgente": tool_call("Triage", {"category": "financeiro", "urgent": True}),
        "texto solto": AIMessage("acho que é sobre um pedido"),
        "categoria inventada": tool_call("Triage", {"category": "vendas", "urgent": False}),
    }
    for label, reply in replies.items():
        chain = build_triage_chain(ScriptedChatModel.from_script([reply]))
        print(f"{label:<20} -> {chain.invoke({'message': 'mensagem'})}")


if __name__ == "__main__":
    main()
