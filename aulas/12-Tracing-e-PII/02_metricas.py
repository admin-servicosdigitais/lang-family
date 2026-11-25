"""Passo 2: medir chamadas, tempo e tokens com um callback."""

import time
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from modelo_falso import ScriptedChatModel


@dataclass
class RunMetrics:
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_seconds: float = 0.0


class MetricsCallback(BaseCallbackHandler):
    def __init__(self) -> None:
        self.metrics = RunMetrics()
        self._started: dict[UUID, float] = {}

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        # run_id separa chamadas que acontecem ao mesmo tempo.
        self._started[run_id] = time.perf_counter()

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        self.metrics.llm_calls += 1
        self.metrics.llm_seconds += time.perf_counter() - self._started.pop(run_id)
        generation = response.generations[0][0]
        if isinstance(generation, ChatGeneration):
            usage = getattr(generation.message, "usage_metadata", None) or {}
            self.metrics.input_tokens += usage.get("input_tokens", 0)
            self.metrics.output_tokens += usage.get("output_tokens", 0)


def slow(messages: list[BaseMessage], tools: list[str]) -> AIMessage:
    time.sleep(0.05)
    return AIMessage("resposta de quatro palavras")


def main() -> None:
    callback = MetricsCallback()
    model = ScriptedChatModel(responder=slow)
    model.batch(["primeira pergunta", "segunda"], {"callbacks": [callback]})
    m = callback.metrics
    print(f"chamadas={m.llm_calls} entrada={m.input_tokens} saída={m.output_tokens}")
    print(f"tempo somado nas chamadas: ao menos 0,1 s? {m.llm_seconds >= 0.1}")
    # Preços fictícios, em dólares por milhão de tokens.
    cost = (m.input_tokens * 1.0 + m.output_tokens * 5.0) / 1_000_000
    print(f"custo estimado (preço fictício): US$ {cost:.6f}")


if __name__ == "__main__":
    main()
