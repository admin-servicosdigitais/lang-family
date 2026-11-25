import pytest
from pydantic import BaseModel, Field

from labs.llm_apis.tests.fakes import FAKES, FakeAPI
from labs.llm_apis.types import Tool


@pytest.fixture(params=sorted(FAKES))
def fake(request: pytest.FixtureRequest) -> FakeAPI:
    """Cada teste que usa esta fixture roda uma vez para cada provedor."""
    return FAKES[request.param]()


class OrderArgs(BaseModel):
    order_id: str = Field(description="Número do pedido, ex.: P-1001")


ORDERS = {"P-1001": "enviado", "P-1002": "aguardando pagamento"}


@pytest.fixture
def order_tool() -> Tool:
    return Tool(
        name="consultar_pedido",
        description="Consulta o status de um pedido.",
        args_model=OrderArgs,
        function=lambda order_id: ORDERS[order_id],
    )
