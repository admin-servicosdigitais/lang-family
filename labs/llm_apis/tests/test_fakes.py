"""O simulador também é testado: um fake permissivo faz testes passarem sem provar nada."""

import os

import pytest

from labs.llm_apis.tests.fakes import FakeAnthropic, FakeOpenAI, assert_all_used
from labs.llm_apis.types import UnavailableError
from labs.shared.settings import LIVE_ONLY_VARS


def test_unexpected_request_is_reported_even_if_the_sdk_swallows_it() -> None:
    fake = FakeAnthropic()  # nenhuma resposta programada

    # O SDK converte a falha do transporte em erro de conexão.
    with pytest.raises(UnavailableError):
        fake.provider(max_retries=0).generate("oi")

    with pytest.raises(AssertionError, match=r"requisição inesperada: /v1/messages"):
        assert_all_used([fake])
    fake.unexpected.clear()  # esperada aqui: não deve reprovar este teste


def test_unused_reply_is_reported() -> None:
    fake = FakeOpenAI()
    fake.queue = [fake.text("usada"), fake.text("sobrou")]

    fake.provider().generate("oi")

    with pytest.raises(AssertionError, match="1 resposta programada e não usada"):
        assert_all_used([fake])
    fake.queue.clear()


def test_offline_run_has_no_credentials(request: pytest.FixtureRequest) -> None:
    if request.config.getoption("--live"):
        pytest.skip("com --live as chaves são necessárias")

    assert [name for name in LIVE_ONLY_VARS if name in os.environ] == []
