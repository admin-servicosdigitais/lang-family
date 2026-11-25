"""Transportes reais (stdio e Streamable HTTP) e um agente consumindo o servidor."""

import socket
import subprocess
import sys
import time
from collections.abc import Iterator

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from mcp import ClientSession
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from labs.mcp_lab.agent import ROOT, build_mcp_agent, stdio_server_params
from labs.shared.fake_model import ScriptedChatModel, tool_call


async def test_stdio_transport_end_to_end() -> None:
    async with (
        stdio_client(stdio_server_params()) as (read, write),
        ClientSession(read, write) as session,
    ):
        info = await session.initialize()
        result = await session.call_tool("consultar_pedido", {"order_id": "P-1002"})

    assert info.serverInfo.name == "loja-aurora"
    assert result.structuredContent is not None
    assert result.structuredContent["status"] == "aguardando pagamento"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_port(port: int, process: subprocess.Popen[bytes], timeout: float = 20) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        assert process.poll() is None, "o servidor encerrou antes de aceitar conexões"
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise AssertionError("o servidor HTTP não subiu a tempo")


@pytest.fixture
def http_server() -> Iterator[str]:
    port = _free_port()
    command = [sys.executable, "-m", "labs.mcp_lab.server"]
    command += ["--transport", "streamable-http", "--port", str(port)]
    process = subprocess.Popen(
        command, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        _wait_for_port(port, process)
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        process.terminate()
        process.wait(timeout=10)


async def test_streamable_http_transport_end_to_end(http_server: str) -> None:
    async with (
        streamable_http_client(http_server) as (read, write, session_id),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        tools = (await session.list_tools()).tools
        result = await session.call_tool("consultar_pedido", {"order_id": "P-1001"})
        assert session_id()

    assert {t.name for t in tools} == {"consultar_pedido", "abrir_chamado"}
    assert not result.isError


async def test_agent_discovers_and_uses_mcp_tools() -> None:
    model = ScriptedChatModel.from_script(
        [
            tool_call("consultar_pedido", {"order_id": "P-1001"}),
            tool_call("consultar_pedido", {"order_id": "P-0"}),
            AIMessage("P-1001 foi enviado; P-0 não existe."),
        ]
    )

    async with (
        stdio_client(stdio_server_params()) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        agent = await build_mcp_agent(session, model)
        state = await agent.ainvoke({"messages": [("human", "status de P-1001 e P-0")]})

    ok, failed = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert "enviado" in ok.text
    assert failed.status == "error"
    assert "não encontrado" in failed.text
    assert state["messages"][-1].text.startswith("P-1001 foi enviado")
