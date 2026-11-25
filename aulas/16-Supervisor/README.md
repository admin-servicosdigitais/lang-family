# Aula 16: Supervisor-Driven Network

**Origem:**
[`langgraph_patterns/supervisor.py`](../../labs/langgraph_patterns/supervisor.py)

## O problema

Um único agente com muitas ferramentas e instruções fica confuso e difícil de controlar.
No padrão **supervisor**, um agente coordena especialistas: escolhe quem age, lê o
relatório e decide o próximo passo.

```
START ──> supervisor ──> trabalhador ──> supervisor ──> ... ──> END
```

| Passo | Arquivo | Conceito novo |
|---|---|---|
| 1 | [`01_roteamento.py`](01_roteamento.py) | `Command` |
| 2 | [`02_decisao_estruturada.py`](02_decisao_estruturada.py) | Schema montado em tempo de execução |
| 3 | [`03_limites_e_falhas.py`](03_limites_e_falhas.py) | Orçamento de passos e falhas |
| 4 | [`04_aprovacao_humana.py`](04_aprovacao_humana.py) | `interrupt` |
| 5 | [`05_exemplo_do_lab.py`](05_exemplo_do_lab.py) | O código do laboratório |

```bash
uv run python aulas/16-Supervisor/01_roteamento.py
```

---

## Passo 1: `Command`

Um nó devolve **para onde ir** (`goto`) e **o que mudar** (`update`) no mesmo objeto. Na
aula 15 a rota era uma aresta condicional separada. A anotação `Command[Literal[...]]`
informa ao grafo os destinos possíveis.

## Passo 2: a decisão

`Literal` montado com os nomes da equipe vira um `enum` no schema. Um nome fora da lista
é rejeitado na validação.

## Passo 3: quem garante é o código

| Risco | Proteção no código |
|---|---|
| Supervisor que nunca termina | `max_steps` |
| Rota inventada | validação e status `decisao_invalida` |
| Trabalhador que falha | relatório com `ok=False`; o supervisor decide |

## Passo 4: aprovação humana

`interrupt` pausa o grafo e devolve um valor a quem chamou. O `checkpointer` guarda o
estado, e `Command(resume=...)` retoma, na mesma `thread_id`. Só o valor `True` aprova:
um texto como `"sim"` é recusa.

**Ao retomar, o nó executa de novo desde o início.** Efeitos antes do `interrupt` se
repetem.

## Passo 5: o exemplo do laboratório

- `Worker` é nome, descrição, função e a marca `needs_approval`. `llm_worker` cria um
  trabalhador que é um agente com ferramentas.
- O supervisor vê a tarefa e os relatórios, não o raciocínio interno dos trabalhadores.
- Os status são `concluido`, `orcamento_esgotado` e `decisao_invalida`.
- A recusa humana vira relatório, e o supervisor segue.

Correções feitas no laboratório antes desta aula:

- Uma pausa (`interrupt`) dentro de um trabalhador era capturada como falha, porque o
  LangGraph a sinaliza por exceção. Agora `GraphBubbleUp` sobe.
- Um trabalhador chamado `FINALIZAR` nunca seria acionado; o nome agora é recusado.

## Limitações que continuam no laboratório

- Uma segunda tarefa na mesma `thread_id` herda passos e relatórios, e pode terminar por
  orçamento esgotado (o exercício abaixo).
- Aprovação sem `checkpointer` pausa, mas não há como retomar.
- Quem aprova não pode ajustar a instrução, só aceitar ou recusar.
- Um trabalhador por vez, sem paralelismo.
- Cada passo custa ao menos uma chamada do supervisor, além das do trabalhador.

## Onde o laboratório comprova isso

Em [`test_supervisor.py`](../../labs/langgraph_patterns/tests/test_supervisor.py):

- `test_routes_workers_based_on_their_reports`
- `test_step_budget_stops_a_supervisor_that_never_finishes`
- `test_route_outside_the_roster_is_rejected`
- `test_worker_failure_is_reported_and_the_team_continues`
- `test_sensitive_worker_waits_for_human_approval`
- `test_interrupt_raised_inside_a_worker_pauses_the_team`
- `test_worker_cannot_use_the_reserved_finish_name`

```bash
uv run pytest labs/langgraph_patterns/tests/test_supervisor.py
```

## Perguntas de entrevista

1. Quando vários agentes são melhores que um? Qual o custo?
2. Que outros arranjos existem além do supervisor?
3. O que o código precisa garantir, independentemente do modelo?
4. Como funciona a pausa para aprovação? O que é preciso guardar?
5. Que cuidado ter com efeitos colaterais ao retomar?
6. Como depurar uma decisão errada do supervisor?

## Exercício

Faça uma nova tarefa na mesma conversa começar com orçamento e relatórios zerados, sem
quebrar a retomada de uma aprovação. Escreva o teste antes.
