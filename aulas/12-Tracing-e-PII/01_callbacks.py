"""Passo 1: callbacks são avisados de cada etapa de uma execução."""

from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate
from modelo_falso import ScriptedChatModel


class TreePrinter(BaseCallbackHandler):
    """Mostra quem chamou quem: é a base de uma árvore de trace."""

    def __init__(self) -> None:
        self.depth: dict[UUID, int] = {}

    def _show(self, text: str, run_id: UUID, parent: UUID | None) -> None:
        self.depth[run_id] = self.depth.get(parent, -1) + 1 if parent else 0
        print(f"{'  ' * self.depth[run_id]}{text}")

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._show(f"início de etapa: {kwargs.get('name')}", run_id, parent_run_id)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._show(f"modelo recebeu: {messages[0][-1].text}", run_id, parent_run_id)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        print(f"    modelo respondeu: {response.generations[0][0].text}")


def main() -> None:
    prompt = ChatPromptTemplate.from_messages([("human", "{question}")])
    chain = (prompt | ScriptedChatModel.from_script([AIMessage("Foi enviado.")])).with_config(
        run_name="atendimento"
    )
    chain.invoke({"question": "Cadê o pedido?"}, {"callbacks": [TreePrinter()]})
    print("\nNinguém mudou a cadeia: o callback vai na configuração da chamada.")


if __name__ == "__main__":
    main()
