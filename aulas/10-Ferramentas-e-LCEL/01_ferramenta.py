"""Passo 1: transformar uma função em ferramenta com ``@tool``."""

import json

from langchain_core.tools import tool


@tool
def calcular_frete(total: float, regiao: str) -> str:
    """Calcula o frete a partir do valor do pedido e da região do país."""
    if total > 299 and regiao.strip().lower() in {"sul", "sudeste"}:
        return "frete grátis"
    return "frete: R$ 29,90"


def main() -> None:
    # Tudo o que o modelo recebe sai da própria função.
    print(f"nome      : {calcular_frete.name}")
    print(f"descrição : {calcular_frete.description}")
    print(f"argumentos: {json.dumps(calcular_frete.args, ensure_ascii=False)}\n")

    print(calcular_frete.invoke({"total": 350, "regiao": "Sudeste"}))
    print(calcular_frete.invoke({"total": "100", "regiao": "sul"}), "(o texto '100' vira número)")
    try:
        calcular_frete.invoke({"total": "muito", "regiao": "sul"})
    except ValueError as error:
        print(f"inválido: {type(error).__name__}")


if __name__ == "__main__":
    main()
