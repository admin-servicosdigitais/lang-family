"""Passo 5: a observabilidade do laboratório (``labs/observability/tracing.py``).

O ``main`` é só para a aula e não envia nada a nenhum serviço.
"""

import os
import re
import time
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.tracers.langchain import LangChainTracer
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
from langsmith import Client as LangSmithClient
from modelo_falso import ScriptedChatModel
from pydantic import BaseModel

# --- Do laboratório: observability/tracing.py ---------------------------------------


_PII_PATTERNS = [
    # O CPF não pode estar colado a letras ou dígitos; o sublinhado separa ("cpf_123...").
    (re.compile(r"(?<![^\W_])\d{3}\.?\d{3}\.?\d{3}-?\d{2}(?![^\W_])"), "[CPF]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[EMAIL]"),
]


def redact(value: Any) -> Any:
    """Substitui CPF e e-mail em qualquer estrutura antes de ela sair do processo.

    Falha fechada: um tipo desconhecido vira texto mascarado em vez de passar intacto
    (o LangGraph, por exemplo, coloca objetos ``Command`` nas saídas dos nós).
    """
    if isinstance(value, str):
        for pattern, replacement in _PII_PATTERNS:
            value = pattern.sub(replacement, value)
        return value
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, dict):
        # Só chaves de texto: outras poderiam virar tipos que não servem de chave.
        return {
            redact(key) if isinstance(key, str) else key: redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list | tuple | set):
        return [redact(item) for item in value]
    if isinstance(value, BaseMessage):
        # Mantém a classe, para a interface continuar mostrando o papel da mensagem.
        names = type(value).model_fields
        return value.model_copy(update={n: redact(getattr(value, n)) for n in names})
    if isinstance(value, BaseModel):
        return redact(value.model_dump())
    if is_dataclass(value) and not isinstance(value, type):
        return redact({f.name: getattr(value, f.name) for f in fields(value)})
    return redact(str(value))


def _mask(*, data: Any, **_: Any) -> Any:
    return redact(data)


def langsmith_client(**kwargs: Any) -> LangSmithClient:
    # O anonymizer cobre entradas, saídas, metadados e também mensagens de erro.
    return LangSmithClient(anonymizer=redact, **kwargs)


def langsmith_tracer(client: LangSmithClient, project: str = "lang-family") -> LangChainTracer:
    return LangChainTracer(client=client, project_name=project)


def langfuse_client(**kwargs: Any) -> Langfuse:
    """Cliente Langfuse; sem argumentos, lê LANGFUSE_* do ambiente."""
    return Langfuse(mask=_mask, **kwargs)


def _redact_error(error: BaseException) -> BaseException:
    message = str(error)
    cleaned = redact(message)
    # Sem PII, a exceção original segue: o tipo importa (ex.: interrupções do LangGraph).
    if cleaned == message:
        return error
    return RuntimeError(f"{type(error).__name__}: {cleaned}")


class RedactingLangfuseHandler(LangfuseCallbackHandler):
    """O ``mask`` do Langfuse não alcança a mensagem de status dos erros, que é
    gravada direto no span; por isso o erro é limpo antes de chegar ao handler."""

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> Any:
        return super().on_llm_error(_redact_error(error), **kwargs)

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> Any:
        return super().on_chain_error(_redact_error(error), **kwargs)

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> Any:
        return super().on_tool_error(_redact_error(error), **kwargs)

    def on_retriever_error(self, error: BaseException, **kwargs: Any) -> Any:
        return super().on_retriever_error(_redact_error(error), **kwargs)

    def on_tool_end(self, output: Any, **kwargs: Any) -> Any:
        return super().on_tool_end(redact(output), **kwargs)


def langfuse_handler(public_key: str | None = None) -> LangfuseCallbackHandler:
    return RedactingLangfuseHandler(public_key=public_key)


def callbacks_from_env() -> list[BaseCallbackHandler]:
    """Liga cada backend de observabilidade que estiver configurado no ambiente."""
    callbacks: list[BaseCallbackHandler] = []
    if os.getenv("LANGSMITH_API_KEY"):
        callbacks.append(langsmith_tracer(langsmith_client()))
    if os.getenv("LANGFUSE_PUBLIC_KEY"):
        langfuse_client()
        callbacks.append(langfuse_handler(os.environ["LANGFUSE_PUBLIC_KEY"]))
    return callbacks


@dataclass
class RunMetrics:
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_seconds: float = 0.0
    tool_calls: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class MetricsCallback(BaseCallbackHandler):
    """Métricas agregadas em memória, independentes de qualquer serviço."""

    def __init__(self) -> None:
        self.metrics = RunMetrics()
        self._started: dict[UUID, float] = {}

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], *, run_id: UUID, **kwargs: Any
    ) -> None:
        # Modelos de texto puro não disparam on_chat_model_start.
        self._started[run_id] = time.perf_counter()

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._started[run_id] = time.perf_counter()

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        self.metrics.llm_calls += 1
        self.metrics.llm_seconds += time.perf_counter() - self._started.pop(run_id)
        generation = response.generations[0][0]
        if isinstance(generation, ChatGeneration):
            usage = getattr(generation.message, "usage_metadata", None) or {}
            self.metrics.input_tokens += usage.get("input_tokens", 0)
            self.metrics.output_tokens += usage.get("output_tokens", 0)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._started.pop(run_id, None)
        self.metrics.errors.append(f"llm: {error}")

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> None:
        self.metrics.tool_calls.append(serialized["name"])

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        # Erros tratados pela ferramenta chegam como mensagem, não como exceção.
        if getattr(output, "status", None) == "error":
            self.metrics.errors.append(f"tool: {output.content}")

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        self.metrics.errors.append(f"tool: {error}")


# --- Somente da aula ----------------------------------------------------------------


def main() -> None:
    message = HumanMessage("Sou ana@exemplo.com, CPF 123.456.789-09")
    print("redact:", redact({"mensagens": [message]})["mensagens"][0].text)
    error = ValueError("cadastro de ana@exemplo.com inválido")
    print("erro  :", _redact_error(error))
    clean = KeyError("sem dado pessoal")
    print("erro sem PII mantém o objeto original?", _redact_error(clean) is clean)

    metrics = MetricsCallback()
    model = ScriptedChatModel.from_script([AIMessage("tudo certo")])
    model.invoke("status do pedido", {"callbacks": [metrics]})
    m = metrics.metrics
    print(f"chamadas={m.llm_calls} tokens={m.input_tokens}/{m.output_tokens} erros={m.errors}")


if __name__ == "__main__":
    main()
