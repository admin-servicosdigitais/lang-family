# Aula 11: agente com LangChain

**Origem:** [`observability/agent.py`](../../labs/observability/agent.py) e
[`shared/fake_model.py`](../../labs/shared/fake_model.py)

## O problema

Numa cadeia, a ordem das etapas é fixa. Num **agente**, o modelo decide quais
ferramentas chamar, e quantas vezes, até ter a resposta. Isso dá flexibilidade e exige
limites.

| Passo | Arquivo | Conceito novo |
|---|---|---|
| 1 | [`01_laco_manual.py`](01_laco_manual.py) | O laço do agente |
| 2 | [`02_create_agent.py`](02_create_agent.py) | `create_agent` |
| 3 | [`03_limites.py`](03_limites.py) | *Middleware* de limites |
| 4 | [`04_memoria.py`](04_memoria.py) | Memória por *thread* |
| 5 | [`05_exemplo_do_lab.py`](05_exemplo_do_lab.py) | O agente do laboratório |

```bash
uv run python aulas/11-Agente-LangChain/01_laco_manual.py
```

## O modelo falso

[`modelo_falso.py`](modelo_falso.py) é cópia do `fake_model.py` do laboratório.
`ScriptedChatModel` herda de `BaseChatModel` e delega a resposta a uma função Python,
por isso funciona com agentes, ferramentas, saída estruturada e *callbacks*. Ele testa a
**orquestração**, não a qualidade das decisões de um modelo.

---

## Passo 1: o laço

Modelo, pedidos de ferramenta, resultados, modelo de novo. Para quando a resposta não
pede ferramenta. É o mesmo laço da aula 03, com as mensagens do LangChain.

## Passo 2: `create_agent`

Monta um grafo do LangGraph com dois nós, `model` e `tools`. O estado é a lista de
mensagens. O resultado de `invoke` traz o histórico completo.

## Passo 3: limites

| *Middleware* | Ao estourar, no laboratório |
|---|---|
| `ModelCallLimitMiddleware(exit_behavior="end")` | encerra o agente com uma mensagem |
| `ToolCallLimitMiddleware` | bloqueia a ferramenta com um `ToolMessage` de erro; o agente segue |

O limite de modelo é o que garante o fim: só o de ferramentas deixaria o modelo
continuar pedindo.

## Passo 4: memória

O `checkpointer` grava o estado ao fim de cada passo, separado por `thread_id`. Na
chamada seguinte da mesma conversa, o histórico é recarregado. `InMemorySaver` se perde
ao encerrar o processo; em produção usa-se um banco.

## Passo 5: o exemplo do laboratório

`build_agent` recebe o modelo e o `checkpointer`, define as instruções, os limites (6
chamadas ao modelo e 5 a ferramentas por execução) e o nome do agente, que aparece nos
traces.

Nenhuma correção foi necessária no laboratório para esta aula.

## Limitações que continuam no laboratório

- Ao estourar o limite, o cliente recebe `Model call limits exceeded: run limit (6/6)`,
  em inglês e sem sentido para ele (o exercício abaixo).
- Os limites são por execução. Não há limite por conversa nem por custo.
- O histórico cresce sem corte nem resumo, e cada volta fica mais cara.
- `thread_id` não é ligado a um usuário: quem souber o id lê a conversa.
- Não há aprovação humana (ver aula 16).

## Onde o laboratório comprova isso

Em [`test_agent.py`](../../labs/observability/tests/test_agent.py):

- `test_agent_chains_tools_and_feeds_results_back`
- `test_tool_error_returns_to_model_instead_of_crashing`
- `test_model_call_limit_stops_a_looping_agent`
- `test_tool_call_limit_blocks_extra_calls`
- `test_memory_is_isolated_by_thread`

```bash
uv run pytest labs/observability/tests/test_agent.py
```

## Perguntas de entrevista

1. Qual a diferença entre uma cadeia e um agente? Quando você evitaria um agente?
2. Como impedir um agente de entrar em laço?
3. O que é *middleware* e que outros usos teria?
4. Como funciona a memória? Que problemas aparecem em conversas longas?
5. Como testar um agente? O que o modelo falso não cobre?
6. Uma ferramenta falha de forma intermitente. Onde tratar?

## Exercício

Faça o agente responder em português e encaminhar para um atendente quando estourar o
limite, escrevendo o teste antes.
