"""Passo 4: um servidor HTTP local, para quando não dá para injetar o transporte.

uv run pytest aulas/04-Testes-sem-Chave/04_servidor_http_local.py -v
"""

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

import openai
import pytest

# --- Do laboratório: shared/fake_http.py ------------------------------------------

# (método, caminho sem query string, corpo JSON ou None) -> (status, corpo JSON)
Handler = Callable[[str, str, Any], tuple[int, Any]]


@contextmanager
def fake_server(handle: Handler) -> Iterator[str]:
    """Sobe o servidor em uma porta livre e devolve a URL base."""

    class _RequestHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _dispatch(self) -> None:
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            body = json.loads(raw) if raw else None
            status, payload = handle(self.command, urlsplit(self.path).path, body)
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_GET = do_POST = do_PATCH = do_PUT = do_DELETE = _dispatch

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), _RequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


# --- Somente da aula --------------------------------------------------------------


def create_ticket(base_url: str, order_id: str) -> str:
    """Cliente que só aceita uma URL: não há transporte para trocar."""
    data = json.dumps({"order_id": order_id}).encode()
    request = urllib.request.Request(f"{base_url}/tickets", data=data, method="POST")
    with urllib.request.urlopen(request, timeout=5) as response:
        return str(json.load(response)["ticket_id"])


def test_client_that_only_accepts_a_url() -> None:
    received: list[tuple[str, str, Any]] = []

    def handle(method: str, path: str, body: Any) -> tuple[int, Any]:
        received.append((method, path, body))
        return 200, {"ticket_id": "C-0001"}

    with fake_server(handle) as url:
        assert create_ticket(url, "P-1001") == "C-0001"

    assert received == [("POST", "/tickets", {"order_id": "P-1001"})]


def test_error_status_travels_through_real_http() -> None:
    with (
        fake_server(lambda *_: (503, {"error": "fora do ar"})) as url,
        pytest.raises(urllib.error.HTTPError, match="503"),
    ):
        create_ticket(url, "P-1001")


def test_sdk_also_works_by_changing_the_base_url() -> None:
    paths: list[str] = []

    def handle(method: str, path: str, body: Any) -> tuple[int, Any]:
        paths.append(path)
        return 401, {"error": {"message": "chave inválida"}}

    with fake_server(handle) as url:
        client = openai.OpenAI(api_key="teste", base_url=f"{url}/v1", max_retries=0)
        with pytest.raises(openai.AuthenticationError):
            client.responses.create(model="modelo", input="oi")

    assert paths == ["/v1/responses"]
