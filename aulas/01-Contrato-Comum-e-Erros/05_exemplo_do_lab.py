"""Passo 5: o contrato completo do laboratório.

Os blocos "do laboratório" reproduzem ``labs/llm_apis/types.py`` e
``labs/llm_apis/base.py``. O provedor falso e o ``main`` são só para a aula.
"""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

# --- Do laboratório: types.py -----------------------------------------------------


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
        )


@dataclass(frozen=True)
class ToolRun:
    name: str
    arguments: dict[str, Any]
    output: str
    is_error: bool = False


@dataclass(frozen=True)
class ChatResult:
    text: str
    provider: str
    model: str
    usage: Usage = field(default_factory=Usage)
    tool_runs: tuple[ToolRun, ...] = ()


@dataclass(frozen=True)
class Tool:
    """Função Python exposta ao modelo; os argumentos são validados por Pydantic."""

    name: str
    description: str
    args_model: type[BaseModel]
    function: Callable[..., object]

    @property
    def json_schema(self) -> dict[str, Any]:
        return self.args_model.model_json_schema()

    def run(self, arguments: dict[str, Any]) -> ToolRun:
        """Executa a ferramenta. Falhas voltam ao modelo em vez de derrubar o fluxo."""
        try:
            args = self.args_model.model_validate(arguments)
            # dict(args) preserva modelos aninhados; model_dump() os viraria em dicts.
            output = str(self.function(**dict(args)))
        except ValidationError as error:
            return ToolRun(self.name, arguments, f"argumentos inválidos: {error}", True)
        except Exception as error:
            return ToolRun(self.name, arguments, f"{type(error).__name__}: {error}", True)
        return ToolRun(self.name, arguments, output)


def run_tool(tools: list[Tool], name: str, arguments: dict[str, Any]) -> ToolRun:
    for tool in tools:
        if tool.name == name:
            return tool.run(arguments)
    return ToolRun(name, arguments, f"ferramenta desconhecida: {name}", True)


class LLMError(Exception):
    """Erro normalizado, independente do SDK que o gerou."""

    retryable = False

    def __init__(self, provider: str, message: str) -> None:
        super().__init__(f"[{provider}] {message}")
        self.provider = provider


class AuthError(LLMError):
    pass


class BadRequestError(LLMError):
    pass


class RateLimitError(LLMError):
    retryable = True


class UnavailableError(LLMError):
    """Erro 5xx, sobrecarga, timeout ou falha de conexão."""

    retryable = True


class RefusalError(LLMError):
    pass


class ToolLoopError(LLMError):
    pass


def error_for_status(provider: str, status: int, message: str) -> LLMError:
    if status in (401, 403):
        return AuthError(provider, message)
    if status == 429:
        return RateLimitError(provider, message)
    if status >= 500 or status == 408:
        return UnavailableError(provider, message)
    return BadRequestError(provider, message)


# --- Do laboratório: base.py ------------------------------------------------------


class LLMProvider(Protocol):
    name: str
    model: str

    def generate(self, prompt: str, *, system: str | None = None) -> ChatResult: ...

    def stream(self, prompt: str, *, system: str | None = None) -> Iterator[str]: ...

    def structured[T: BaseModel](
        self, prompt: str, schema: type[T], *, system: str | None = None
    ) -> T: ...

    def run_tools(
        self,
        prompt: str,
        tools: list[Tool],
        *,
        system: str | None = None,
        max_turns: int = 5,
    ) -> ChatResult: ...


class FallbackRouter:
    """Tenta os provedores em ordem, avançando só em falhas transitórias.

    Erros de autenticação ou de requisição sobem imediatamente: trocar de provedor
    esconderia um defeito de configuração ou de código.
    """

    def __init__(self, providers: Sequence[LLMProvider]) -> None:
        if not providers:
            raise ValueError("informe ao menos um provedor")
        self.providers = list(providers)
        self.failures: list[LLMError] = []

    def generate(self, prompt: str, *, system: str | None = None) -> ChatResult:
        last_error: LLMError | None = None
        for provider in self.providers:
            try:
                return provider.generate(prompt, system=system)
            except LLMError as error:
                if not error.retryable:
                    raise
                self.failures.append(error)
                last_error = error
        assert last_error is not None
        raise last_error


# --- Somente da aula: provedor falso e demonstração -------------------------------


class FakeProvider:
    """Cumpre o contrato inteiro com respostas fixas. Nenhum modelo é chamado."""

    model = "modelo-falso"

    def __init__(self, name: str, failure: LLMError | None = None) -> None:
        self.name = name
        self.failure = failure

    def generate(self, prompt: str, *, system: str | None = None) -> ChatResult:
        if self.failure:
            raise self.failure
        return ChatResult(f"resposta para: {prompt}", self.name, self.model, Usage(10, 5))

    def stream(self, prompt: str, *, system: str | None = None) -> Iterator[str]:
        yield from self.generate(prompt, system=system).text.split()

    def structured[T: BaseModel](
        self, prompt: str, schema: type[T], *, system: str | None = None
    ) -> T:
        return schema.model_validate({"order_id": "P-1001"})

    def run_tools(
        self,
        prompt: str,
        tools: list[Tool],
        *,
        system: str | None = None,
        max_turns: int = 5,
    ) -> ChatResult:
        # Finge que o modelo pediu a primeira ferramenta. O laço real é o tema da aula 3.
        run = run_tool(tools, tools[0].name, {"order_id": "P-1001"})
        text = f"A ferramenta devolveu: {run.output}"
        return ChatResult(text, self.name, self.model, Usage(20, 10), (run,))


class OrderArgs(BaseModel):
    order_id: str


def order_status(order_id: str) -> str:
    return {"P-1001": "enviado"}[order_id]


def main() -> None:
    tool = Tool("consultar_pedido", "Consulta o status de um pedido.", OrderArgs, order_status)

    print("Schema enviado ao modelo:", tool.json_schema)

    print("\nExecuções de ferramenta (nenhuma lança exceção):")
    attempts: list[tuple[str, dict[str, Any]]] = [
        ("consultar_pedido", {"order_id": "P-1001"}),
        ("consultar_pedido", {}),
        ("consultar_pedido", {"order_id": "P-9999"}),
        ("apagar_tudo", {}),
    ]
    for name, arguments in attempts:
        run = run_tool([tool], name, arguments)
        first_line = run.output.splitlines()[0]
        print(f"  is_error={run.is_error!s:<5} {name} {arguments} -> {first_line}")

    print("\nContrato completo:")
    provider: LLMProvider = FakeProvider("openai")
    print("  generate  :", provider.generate("oi").text)
    print("  stream    :", list(provider.stream("oi")))
    print("  structured:", provider.structured("qual pedido?", OrderArgs))
    print("  run_tools :", provider.run_tools("status do P-1001", [tool]).text)

    print("\nFallback:")
    outage = error_for_status("openai", 503, "indisponível")
    router = FallbackRouter([FakeProvider("openai", outage), FakeProvider("anthropic")])
    print("  respondeu :", router.generate("oi").provider)
    print("  falhas    :", [str(failure) for failure in router.failures])


if __name__ == "__main__":
    main()
