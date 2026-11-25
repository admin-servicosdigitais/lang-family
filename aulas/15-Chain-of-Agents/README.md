# Aula 15: Chain-of-Agents

**Origem:**
[`langgraph_patterns/chain_of_agents.py`](../../labs/langgraph_patterns/chain_of_agents.py),
baseado em Zhang et al. (2024)

## O problema

Um documento longo pode não caber no contexto. Cortá-lo perde informação, e a busca por
trechos (RAG) falha quando a resposta exige ligar fatos distantes. No
**Chain-of-Agents**, trabalhadores leem um trecho cada, em sequência, e passam adiante
uma **unidade de comunicação** (anotações). Um gerente responde só com as anotações.

```
trabalhador(trecho 1) ─CU─> trabalhador(trecho 2) ─CU─> ... ─CU─> gerente ─> resposta
```

| Passo | Arquivo | Conceito novo |
|---|---|---|
| 1 | [`01_problema_do_contexto.py`](01_problema_do_contexto.py) | O limite de contexto |
| 2 | [`02_cadeia_em_python.py`](02_cadeia_em_python.py) | A ideia, sem framework |
| 3 | [`03_primeiro_grafo.py`](03_primeiro_grafo.py) | Estado, nós, arestas e redutores |
| 4 | [`04_arestas_condicionais.py`](04_arestas_condicionais.py) | Laço e limite de recursão |
| 5 | [`05_exemplo_do_lab.py`](05_exemplo_do_lab.py) | O código do laboratório |

```bash
uv run python aulas/15-Chain-of-Agents/01_problema_do_contexto.py
```

O "modelo" é um conjunto de regras. Ele só liga os fatos se as anotações os
transportarem, o que torna o fluxo verificável.

---

## Passos 1 e 2: a ideia

Cada trabalhador vê o seu trecho e a CU anterior. Repare no resultado com a ordem
invertida: o segundo fato passa antes de se saber quem lidera, e se perde. O padrão
**depende da ordem**.

## Passo 3: o grafo

- **Estado:** um `TypedDict` compartilhado.
- **Nó:** função que lê o estado e devolve apenas o que mudou.
- **Aresta:** ordem entre os nós.
- **Redutor:** como combinar o valor novo com o antigo. Sem ele, substitui; com
  `operator.add`, acumula.

## Passo 4: laço

Uma aresta condicional decide o próximo nó a partir do estado, o que permite ciclos.
`recursion_limit` interrompe grafos que não terminam.

## Passo 5: o exemplo do laboratório

- Nós `worker` e `manager`; o trabalhador repete enquanto houver trechos.
- `notes_limit` é imposto **em código**: a CU não cresce com o documento.
- `history` usa redutor e guarda cada versão das anotações, útil para depurar.
- `ask` ajusta o `recursion_limit` à quantidade de trechos.
- O gerente pode usar um modelo diferente dos trabalhadores.

Com `notes_limit=14`, o corte descarta justamente o fato mais recente.

Correções feitas no laboratório antes desta aula: uma resposta vazia de um trabalhador
apagava as anotações acumuladas, e um `notes_limit` zero ou negativo era aceito.

## Comparação

| Abordagem | Força | Fraqueza |
|---|---|---|
| Contexto longo | simples | custo e limite de tamanho |
| RAG | barato por pergunta | perde ligações entre trechos distantes |
| Map-reduce | paralelo | cada leitura ignora as demais |
| Chain-of-Agents | acumula raciocínio | sequencial e lento; depende da ordem |

## Limitações que continuam no laboratório

- O corte por tamanho elimina o fim das anotações (o exercício abaixo).
- Uma chamada por trecho, em sequência: a latência cresce com o documento.
- Uma falha no meio perde o trabalho feito: o grafo não usa `checkpointer`.
- O texto de um trecho pode conter as marcas `</trecho>` e confundir o trabalhador.

## Onde o laboratório comprova isso

Em
[`test_chain_of_agents.py`](../../labs/langgraph_patterns/tests/test_chain_of_agents.py):

- `test_combines_facts_from_distant_chunks`
- `test_each_agent_sees_only_its_chunk_and_the_manager_sees_none`
- `test_communication_unit_is_bounded`
- `test_blank_reply_keeps_the_previous_notes`
- `test_notes_limit_must_be_positive`

```bash
uv run pytest labs/langgraph_patterns/tests/test_chain_of_agents.py
```

## Perguntas de entrevista

1. Quando o Chain-of-Agents é melhor que RAG?
2. O que é um redutor no LangGraph?
3. Como o tamanho da unidade de comunicação afeta o resultado?
4. Qual o custo em chamadas e latência?
5. O que acontece se um fato aparece antes de se saber que ele importa?
6. Como retomar uma execução que falhou no meio?

## Exercício

Quando as anotações passarem do limite, peça ao trabalhador uma versão resumida antes de
cortar. Escreva o teste antes e garanta que o limite continue valendo.
