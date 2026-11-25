"""Escolha do modelo real (LangChain) conforme a chave disponível."""

import os

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from labs.shared.settings import API_KEY_VARS, model_for

_LANGCHAIN_PROVIDERS = {
    "anthropic": "anthropic",
    "openai": "openai",
    "gemini": "google_genai",
}


def available_provider() -> str | None:
    """Provedor definido em LLM_PROVIDER ou o primeiro com chave configurada."""
    forced = os.getenv("LLM_PROVIDER")
    candidates = [forced] if forced else list(_LANGCHAIN_PROVIDERS)
    return next((p for p in candidates if os.getenv(API_KEY_VARS[p])), None)


def live_chat_model() -> BaseChatModel:
    """Modelo real para testes ``live``; pula o teste se não houver chave."""
    provider = available_provider()
    if provider is None:
        import pytest

        pytest.skip(f"defina uma destas chaves: {', '.join(API_KEY_VARS.values())}")
    return init_chat_model(model_for(provider), model_provider=_LANGCHAIN_PROVIDERS[provider])
