"""Os mesmos recursos contra as APIs reais: ``uv run pytest --live labs/llm_apis``."""

from collections.abc import Callable

import pytest
from pydantic import BaseModel

from labs.llm_apis.anthropic_provider import AnthropicProvider
from labs.llm_apis.base import LLMProvider
from labs.llm_apis.gemini_provider import GeminiProvider
from labs.llm_apis.openai_provider import OpenAIProvider
from labs.llm_apis.types import Tool
from labs.shared.settings import API_KEY_VARS, require_env

pytestmark = pytest.mark.live

PROVIDERS: dict[str, Callable[[], LLMProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
}


@pytest.fixture(params=sorted(PROVIDERS))
def provider(request: pytest.FixtureRequest) -> LLMProvider:
    require_env(API_KEY_VARS[request.param])
    return PROVIDERS[request.param]()


class City(BaseModel):
    name: str
    country: str


def test_generate(provider: LLMProvider) -> None:
    result = provider.generate("Qual é a capital do Brasil? Responda só o nome.")

    assert "bras" in result.text.lower()
    assert result.usage.input_tokens > 0


def test_stream(provider: LLMProvider) -> None:
    text = "".join(provider.stream("Conte de 1 a 5, separando por vírgulas."))

    assert "3" in text


def test_structured(provider: LLMProvider) -> None:
    city = provider.structured("Em que país fica Lisboa? Preencha cidade e país.", City)

    assert "lisbo" in city.name.lower()
    assert "portugal" in city.country.lower()


def test_tool_calling(provider: LLMProvider, order_tool: Tool) -> None:
    result = provider.run_tools("Use a ferramenta e diga o status do pedido P-1002.", [order_tool])

    assert [r.name for r in result.tool_runs] == ["consultar_pedido"]
    assert "pagamento" in result.text.lower()
