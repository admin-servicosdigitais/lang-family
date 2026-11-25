"""Configuração global dos testes.

Três níveis de teste:
- padrão: offline e determinístico, sem chaves e sem Docker;
- ``docker``: roda se os contêineres de ``infra/`` estiverem no ar, senão é pulado;
- ``live``: só roda com ``--live`` (evita custo acidental) e com a chave do provedor.
"""

import os

import pytest
from dotenv import load_dotenv

from labs.shared.settings import LIVE_ONLY_VARS

load_dotenv()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--live", action="store_true", help="executa testes contra APIs reais")


def pytest_configure(config: pytest.Config) -> None:
    if not config.getoption("--live"):
        # Um .env com tracing ligado não pode fazer testes offline enviarem dados.
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        # Um teste que esqueça o simulador falha por falta de chave, em vez de gerar custo.
        for name in LIVE_ONLY_VARS:
            os.environ.pop(name, None)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--live"):
        return
    skip = pytest.mark.skip(reason="teste real: use --live (pode gerar custo)")
    for item in items:
        if item.get_closest_marker("live"):
            item.add_marker(skip)
