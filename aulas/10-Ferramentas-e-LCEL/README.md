# Aula 10: ferramentas e LCEL

**Origem:** [`observability/tools.py`](../../labs/observability/tools.py) e
[`observability/triage.py`](../../labs/observability/triage.py)

## O problema

Na aula 03 o laço de ferramentas foi escrito à mão, uma vez por provedor. O LangChain
oferece peças comuns a todos: ferramentas declaradas a partir de funções e **cadeias**
compostas com LCEL (*LangChain Expression Language*).

| Passo | Arquivo | Conceito novo |
|---|---|---|
| 1 | [`01_ferramenta.py`](01_ferramenta.py) | `@tool` |
| 2 | [`02_erros_de_ferramenta.py`](02_erros_de_ferramenta.py) | `ToolException` |
| 3 | [`03_lcel.py`](03_lcel.py) | Composição com `\|` |
| 4 | [`04_saida_estruturada_e_roteamento.py`](04_saida_estruturada_e_roteamento.py) | Classificar e rotear |
| 5 | [`05_exemplo_do_lab.py`](05_exemplo_do_lab.py) | O código do laboratório |

```bash
uv run python aulas/10-Ferramentas-e-LCEL/01_ferramenta.py
```

[`modelo_falso.py`](modelo_falso.py) é um arquivo de apoio, cópia do modelo roteirizado
do laboratório (explicado na aula 11).

---

## Passo 1: `@tool`

O nome vem da função, a descrição da *docstring* e o schema das anotações de tipo. A
*docstring* é parte do prompt. Os argumentos são validados pelo Pydantic, que converte
`"100"` em número e rejeita `"muito"`.

## Passo 2: erros

- Chamada com um `ToolCall`, a ferramenta devolve um `ToolMessage`, com `status`.
- `ToolException` representa um erro **previsto**. Com `handle_tool_error = True`, vira
  mensagem para o modelo.
- Outras exceções continuam subindo: um defeito não deve ser escondido.

## Passo 3: LCEL

Cada peça é um `Runnable`, com a mesma interface: `invoke`, `batch`, `stream` e as
versões assíncronas. O operador `|` liga a saída de uma à entrada da próxima, e a cadeia
inteira ganha a interface.

LCEL serve para fluxos lineares ou com desvios simples. Com ciclos e estado, use
LangGraph (aulas 15 a 17).

## Passo 4: classificar e rotear

- `with_structured_output(Triage)` faz o modelo responder no formato do modelo Pydantic.
- `RunnablePassthrough.assign` acrescenta uma chave sem perder a entrada.
- `RunnableBranch` avalia as condições em ordem; a última opção é o padrão. A urgência
  vem primeiro, então vence a categoria.
- Quem decide a rota é o código. O modelo só classifica.

## Passo 5: o exemplo do laboratório

Duas correções foram feitas no laboratório antes desta aula:

- Quando o modelo respondia sem o formato pedido, a triagem quebrava com
  `AttributeError` ou erro de validação. Agora a mensagem vai para a **fila humana**.
  `with_fallbacks` captura só os erros de formato; falhas de rede ou de chave continuam
  subindo.
- Espaços ao redor da região ou do número do pedido mudavam o resultado: `" Sul "`
  cobrava frete.

## Limitações que continuam no laboratório

- Região desconhecida ou valor negativo são aceitos e cobram frete (o exercício abaixo).
- A regra do frete está escrita na ferramenta e repetida no documento de entregas; podem
  divergir.
- Os dados dos pedidos ficam em memória.
- A mensagem do cliente vai direto ao classificador e pode tentar manipulá-lo.

## Onde o laboratório comprova isso

Em [`test_agent.py`](../../labs/observability/tests/test_agent.py):

- `test_triage_chain_routes_by_structured_output`
- `test_unreadable_classification_goes_to_a_human`
- `test_triage_does_not_hide_a_failing_model`
- `test_tools_ignore_surrounding_spaces`

```bash
uv run pytest labs/observability/tests/test_agent.py
```

## Perguntas de entrevista

1. O que o LangChain acrescenta ao uso direto do SDK? Que custo traz?
2. Como o modelo "sabe" o que uma ferramenta faz?
3. O que é um `Runnable` e por que a composição é útil?
4. Quando usar uma cadeia e quando usar um agente?
5. O que fazer quando a saída estruturada falha?
6. Como medir se a classificação está boa?

## Exercício

Restrinja `regiao` às cinco regiões do país e exija `total` maior ou igual a zero,
escrevendo o teste antes. Decida como o erro chega ao modelo.
