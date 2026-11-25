"""Passo 2: o modelo escolhe a rota, dentro de um schema montado com a equipe."""

from collections.abc import Sequence
from typing import Any, Literal

from modelo_falso import ScriptedChatModel, tool_call
from pydantic import BaseModel, Field, create_model

FINISH = "FINALIZAR"


def decision_schema(workers: Sequence[str]) -> type[BaseModel]:
    options = (*workers, FINISH)
    # As opções só são conhecidas em tempo de execução.
    route: Any = Literal[options]
    schema: type[BaseModel] = create_model(
        "Decision",
        __doc__="Próximo passo escolhido pelo supervisor.",
        next=(route, Field(description="Trabalhador a acionar, ou FINALIZAR")),
        instruction=(str, Field(description="Instrução ao trabalhador")),
    )
    return schema


def main() -> None:
    schema = decision_schema(["pedidos", "redator"])
    print(f"opções permitidas: {schema.model_json_schema()['properties']['next']['enum']}\n")

    for choice in ("pedidos", FINISH, "apagar_tudo"):
        model = ScriptedChatModel.from_script(
            [tool_call("Decision", {"next": choice, "instruction": "..."})]
        )
        router = model.with_structured_output(schema)
        try:
            decision: Any = router.invoke("próximo passo?")
            print(f"{choice:<12} aceito: next={decision.next}")
        except ValueError:  # ValidationError é um ValueError
            print(f"{choice:<12} rejeitado: não está na equipe")


if __name__ == "__main__":
    main()
