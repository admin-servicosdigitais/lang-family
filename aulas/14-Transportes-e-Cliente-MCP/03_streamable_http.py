"""Passo 3: transporte Streamable HTTP, com o servidor ouvindo em uma porta.

Com ``--server PORTA`` este arquivo é o servidor.
"""

import asyncio
import socket
import subprocess
import sys
import time

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.fastmcp import FastMCP


def serve(port: int) -> None:
    # 127.0.0.1: só aceita conexões desta máquina. Não há autenticação.
    server = FastMCP("aula-http", host="127.0.0.1", port=port, log_level="WARNING")

    @server.tool()
    def eco(texto: str) -> str:
        """Devolve o texto."""
        return texto

    server.run("streamable-http")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for(port: int) -> None:
    for _ in range(200):
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise RuntimeError("o servidor não subiu a tempo")


async def client(url: str) -> None:
    async with (
        streamable_http_client(url) as (read, write, session_id),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        result = await session.call_tool("eco", {"texto": "olá"})
        print(f"resultado: {result.structuredContent}")
        print(f"o servidor criou uma sessão? {bool(session_id())}")


def probe(url: str) -> None:
    """Um site malicioso tentando falar com o servidor local pelo navegador."""
    for header, value in (("Host", "malicioso.exemplo"), ("Origin", "http://malicioso.exemplo")):
        response = httpx.post(url, json={}, headers={header: value}, timeout=5)
        print(f"cabeçalho {header} estranho -> HTTP {response.status_code}")


def main() -> None:
    port = free_port()
    process = subprocess.Popen(
        [sys.executable, __file__, "--server", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_for(port)
        url = f"http://127.0.0.1:{port}/mcp"
        asyncio.run(client(url))
        probe(url)
    finally:
        process.terminate()
        process.wait(timeout=10)


if __name__ == "__main__":
    if "--server" in sys.argv:
        serve(int(sys.argv[-1]))
    else:
        main()
