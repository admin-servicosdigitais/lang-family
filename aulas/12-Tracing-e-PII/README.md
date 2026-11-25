# Aula 12: tracing e dados pessoais

**Origem:** [`observability/tracing.py`](../../labs/observability/tracing.py)

## O problema

Quando um agente responde errado, é preciso ver **o que ele fez**: prompts, chamadas de
ferramenta, tempos, tokens e erros. Um **trace** é a árvore dessas etapas. Ele costuma ir
para um serviço externo e leva junto o que o cliente escreveu.

```
execução (trace)
└── agente
    ├── modelo       tokens, tempo
    ├── ferramenta   entrada, saída
    └── modelo
```

| Passo | Arquivo | Conceito novo |
|---|---|---|
| 1 | [`01_callbacks.py`](01_callbacks.py) | Callbacks e árvore de execuções |
| 2 | [`02_metricas.py`](02_metricas.py) | Métricas e custo |
| 3 | [`03_mascaramento.py`](03_mascaramento.py) | Mascaramento de CPF e e-mail |
| 4 | [`04_enviar_traces.py`](04_enviar_traces.py) | Envio ao LangSmith falso |
| 5 | [`05_exemplo_do_lab.py`](05_exemplo_do_lab.py) | O código do laboratório |

```bash
uv run python aulas/12-Tracing-e-PII/01_callbacks.py
```

---

## Passo 1: callbacks

O LangChain avisa os *handlers* no início e no fim de cada etapa. `run_id` e
`parent_run_id` montam a árvore. O callback é passado em `config`, sem mudar a cadeia.

## Passo 2: métricas

Um callback acumula chamadas, tempo e tokens (`usage_metadata`). Guardar o início por
`run_id` permite chamadas simultâneas. O custo sai dos tokens multiplicados pelo preço,
que muda por modelo e entre entrada e saída. Os tokens do modelo falso são contagem de
palavras.

## Passo 3: mascaramento

- Percorre textos, dicionários (chaves e valores) e listas.
- **Falha fechada:** um tipo desconhecido vira texto mascarado em vez de passar intacto.
- Expressão regular tem limites: CPF com espaços ou como número escapa, nomes e
  endereços não são detectados, e um telefone de 11 dígitos é marcado como CPF.

## Passo 4: enviar traces

O cliente do LangSmith aceita um `anonymizer`, aplicado antes do envio. O passo mostra o
mesmo trace com e sem ele, capturado por um servidor local.

## Passo 5: o exemplo do laboratório

| Peça | Papel |
|---|---|
| `redact` | Mascara qualquer estrutura |
| `langsmith_client` | LangSmith com `anonymizer` |
| `langfuse_client` | Langfuse com `mask` |
| `RedactingLangfuseHandler` | Cobre o que o `mask` não alcança: a mensagem de erro |
| `callbacks_from_env` | Liga cada serviço que tiver chave |
| `MetricsCallback` | Métricas locais, sem serviço |

`_redact_error` devolve a **exceção original** quando não há dado pessoal, porque o tipo
importa: o LangGraph usa exceções para interromper o grafo.

| LangSmith | Langfuse |
|---|---|
| Serviço da LangChain, integrado ao ecossistema | Código aberto, pode ser hospedado por você |
| Aqui: `LangChainTracer` | Aqui: `CallbackHandler`, sobre OpenTelemetry |

Correções feitas no laboratório antes desta aula:

- Dados pessoais em **chaves** de dicionário não eram mascarados.
- CPF ao lado de sublinhado, como em `cpf_12345678909`, escapava.
- `MetricsCallback` falhava com modelos de texto puro, que não disparam
  `on_chat_model_start`.

## Limitações que continuam no laboratório

- CPF com espaços ou numérico escapa; telefones geram falso positivo (o exercício).
- CPF colado a letras não é mascarado, para não corromper identificadores.
- `user_id` e `session_id` não passam pelo mascaramento.
- `MetricsCallback.errors` guarda a mensagem sem mascarar.
- Os testes com o servidor Langfuse real exigem os contêineres de `infra/`.

## Onde o laboratório comprova isso

Em [`test_tracing.py`](../../labs/observability/tests/test_tracing.py):

- `test_redact_walks_nested_structures`
- `test_redact_covers_dictionary_keys`
- `test_redact_finds_cpf_next_to_underscore`
- `test_metrics_cover_text_completion_models`
- `test_langsmith_payload_has_no_pii`
- `test_langfuse_error_messages_have_no_pii`

```bash
uv run pytest labs/observability/tests/test_tracing.py
```

## Perguntas de entrevista

1. Qual a diferença entre log, métrica e trace?
2. Que métricas você acompanharia em um agente em produção?
3. Como calcular e controlar o custo?
4. Por que mascarar antes do envio, e não no serviço?
5. Quais os limites do mascaramento por expressão regular?
6. Como garantir que a observabilidade não derrube a aplicação?
7. Você guardaria os prompts completos? Por quanto tempo?

## Exercício

Valide os dígitos verificadores do CPF para reduzir falsos positivos e trate o CPF com
espaços, escrevendo os testes antes. Diga o que acontece com um CPF digitado errado.
