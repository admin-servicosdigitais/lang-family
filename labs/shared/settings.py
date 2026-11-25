"""Modelos padrão, sobrescrevíveis por variável de ambiente (ver ``.env.example``)."""

import os

import pytest

DEFAULT_MODELS = {
    "openai": "gpt-6-luna",
    "anthropic": "claude-haiku-5-5",
    "gemini": "gemini-3.5-flash-lite",
}

API_KEY_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def model_for(provider: str) -> str:
    return os.getenv(f"{provider.upper()}_MODEL") or DEFAULT_MODELS[provider]


def require_env(name: str) -> str:
    """Pula o teste atual quando a variável de ambiente não está definida."""
    value = os.getenv(name)
    if not value:
        pytest.skip(f"defina {name} para executar este teste")
    return value
