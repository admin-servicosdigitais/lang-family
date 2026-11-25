"""Agente de atendimento com LangChain, com limites de execução e memória."""

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from labs.observability.tools import TOOLS

SYSTEM_PROMPT = (
    "Você é o atendimento da Loja Aurora. Use as ferramentas para obter dados de pedidos; "
    "nunca invente status ou valores. Responda em português, de forma curta."
)


def build_agent(
    model: BaseChatModel,
    *,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    max_model_calls: int = 6,
    max_tool_calls: int = 5,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    # Proteções contra laços infinitos (e contas altas).
    limits: list[Any] = [
        ModelCallLimitMiddleware(run_limit=max_model_calls, exit_behavior="end"),
        ToolCallLimitMiddleware(run_limit=max_tool_calls),
    ]
    return create_agent(
        model,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        middleware=limits,
        checkpointer=checkpointer,
        name="atendimento",
    )
