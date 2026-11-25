"""Passo 4: classificar com saída estruturada e rotear a mensagem."""

from typing import Any, Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda, RunnablePassthrough
from modelo_falso import ScriptedChatModel, tool_call
from pydantic import BaseModel


class Triage(BaseModel):
    """Classificação de uma mensagem de cliente."""

    category: Literal["pedido", "outro"]
    urgent: bool


PROMPT = ChatPromptTemplate.from_messages(
    [("system", "Classifique a mensagem do cliente."), ("human", "{message}")]
)


def main() -> None:
    for category, urgent in (("pedido", False), ("outro", False), ("pedido", True)):
        # O modelo "responde" pedindo a ferramenta de nome Triage: é assim que o
        # with_structured_output obtém o formato.
        model = ScriptedChatModel.from_script(
            [tool_call("Triage", {"category": category, "urgent": urgent})]
        )
        classify = PROMPT | model.with_structured_output(Triage)
        route: Runnable[dict[str, Any], str] = RunnableBranch(
            # Pares (condição, destino), avaliados na ordem; o último é o padrão.
            (lambda x: x["triage"].urgent, RunnableLambda(lambda _: "fila-humana")),
            (lambda x: x["triage"].category == "pedido", RunnableLambda(lambda _: "pedidos")),
            RunnableLambda(lambda _: "faq"),
        )
        # assign acrescenta "triage" ao dicionário sem perder a mensagem original.
        enriched = RunnablePassthrough.assign(triage=classify)
        data = {"message": "mensagem"}
        print(f"{category:<7} urgente={urgent!s:<5} -> {(enriched | route).invoke(data)}")


if __name__ == "__main__":
    main()
