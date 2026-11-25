"""Servidor HTTP local mínimo, usado para simular serviços de terceiros nos testes.

Como o cliente fala HTTP de verdade, funciona com qualquer biblioteca de transporte.
"""

import json
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

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
