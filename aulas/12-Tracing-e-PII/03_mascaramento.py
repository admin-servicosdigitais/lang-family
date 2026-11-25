"""Passo 3: mascarar dados pessoais antes de saírem do processo."""

import re
from typing import Any

_PII_PATTERNS = [
    (re.compile(r"(?<![^\W_])\d{3}\.?\d{3}\.?\d{3}-?\d{2}(?![^\W_])"), "[CPF]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[EMAIL]"),
]


class Customer:
    def __init__(self, email: str) -> None:
        self.email = email

    def __repr__(self) -> str:
        return f"Customer({self.email})"


def redact(value: Any) -> Any:
    if isinstance(value, str):
        for pattern, replacement in _PII_PATTERNS:
            value = pattern.sub(replacement, value)
        return value
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, dict):
        return {redact(k) if isinstance(k, str) else k: redact(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [redact(item) for item in value]
    # Falha fechada: tipo desconhecido vira texto mascarado, nunca passa intacto.
    return redact(str(value))


def main() -> None:
    print("mascarado")
    for value in (
        "CPF 123.456.789-09 e ana@exemplo.com",
        {"ana@exemplo.com": ["cpf_12345678909"]},
        Customer("ana@exemplo.com"),
    ):
        print(f"  {value!r} -> {redact(value)!r}")

    print("\nescapa ou engana: expressão regular não é detecção completa")
    for sample in ("123 456 789 09", 12345678909, "Ana Souza, Rua A, 10", "telefone 11987654321"):
        print(f"  {sample!r} -> {redact(sample)!r}")


if __name__ == "__main__":
    main()
