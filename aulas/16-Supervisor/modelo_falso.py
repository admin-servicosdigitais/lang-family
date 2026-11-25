"""Arquivo de apoio: cópia de ``labs/shared/fake_model.py``.

Modelo de chat determinístico: uma função Python decide a resposta. Sem rede.
"""

from collections.abc import Callable, Sequence
from itertools import count
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LangSmithParams, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

Responder = Callable[[list[BaseMessage], list[str]], AIMessage]

_call_ids = count(1)


def tool_call(name: str, args: dict[str, Any]) -> AIMessage:
    """Resposta do modelo pedindo a execução de uma ferramenta."""
    call = {"name": name, "args": args, "id": f"call_{next(_call_ids)}", "type": "tool_call"}
    return AIMessage(content="", tool_calls=[call])


class ScriptedChatModel(BaseChatModel):
    """Delega cada chamada a ``responder(mensagens, nomes_das_ferramentas)``."""

    responder: Responder
    model_name: str = "modelo-roteirizado"
    calls: list[list[BaseMessage]] = Field(default_factory=list)

    @classmethod
    def from_script(cls, replies: Sequence[AIMessage]) -> "ScriptedChatModel":
        """Devolve as respostas na ordem dada; falha se o roteiro acabar."""
        pending = list(replies)

        def responder(_: list[BaseMessage], __: list[str]) -> AIMessage:
            if not pending:
                raise AssertionError("o roteiro do modelo falso acabou")
            return pending.pop(0)

        return cls(responder=responder)

    @property
    def _llm_type(self) -> str:
        return "scripted-chat-model"

    def _get_ls_params(self, stop: list[str] | None = None, **kwargs: Any) -> LangSmithParams:
        # Nome do modelo lido pelas ferramentas de observabilidade.
        return LangSmithParams(
            ls_provider="fake", ls_model_name=self.model_name, ls_model_type="chat"
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        tool_names = [tool["function"]["name"] for tool in kwargs.get("tools", [])]
        reply = self.responder(messages, tool_names).model_copy()
        # Uso de tokens aproximado (palavras), para a observabilidade ter o que medir.
        input_tokens = sum(len(m.text.split()) for m in messages)
        output_tokens = len(reply.text.split()) or 1
        reply.usage_metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def bind_tools(
        self, tools: Sequence[Any], *, tool_choice: str | None = None, **kwargs: Any
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self.bind(tools=[convert_to_openai_tool(tool) for tool in tools], **kwargs)
