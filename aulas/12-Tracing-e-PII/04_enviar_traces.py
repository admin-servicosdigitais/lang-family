"""Passo 4: enviar traces ao LangSmith, aqui um servidor falso, já mascarados."""

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.tracers.langchain import LangChainTracer
from langsmith import Client
from modelo_falso import ScriptedChatModel


@contextmanager
def fake_langsmith(runs: dict[str, dict[str, Any]]) -> Iterator[str]:
    """Servidor local (aula 04) que guarda as execuções recebidas."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _receive(self) -> None:
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if self.path.startswith("/runs") and raw:
                body = json.loads(raw)
                fields = {k: v for k, v in body.items() if v is not None}
                runs.setdefault(body["id"], {}).update(fields)
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        do_GET = do_POST = do_PATCH = _receive

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def hide_email(data: Any) -> Any:
    """Mascarador reduzido, aplicado pelo cliente antes do envio."""
    return json.loads(json.dumps(data, default=str).replace("ana@exemplo.com", "[EMAIL]"))


def send(anonymizer: Any) -> str:
    runs: dict[str, dict[str, Any]] = {}
    with fake_langsmith(runs) as url:
        client = Client(
            api_url=url, api_key="falsa", auto_batch_tracing=False, anonymizer=anonymizer
        )
        tracer = LangChainTracer(client=client, project_name="aula")
        model = ScriptedChatModel.from_script([AIMessage("Aviso enviado a ana@exemplo.com")])
        model.invoke("Meu e-mail é ana@exemplo.com", {"callbacks": [tracer]})
        tracer.wait_for_futures()
    [run] = runs.values()
    print(f"  tipo={run['run_type']} projeto={run['session_name']}")
    return json.dumps([run["inputs"], run["outputs"]], ensure_ascii=False)


def main() -> None:
    for title, anonymizer in (("sem mascarador", None), ("com mascarador", hide_email)):
        print(title)
        payload = send(anonymizer)
        print(f"  e-mail saiu do processo? {'ana@exemplo.com' in payload}")


if __name__ == "__main__":
    main()
