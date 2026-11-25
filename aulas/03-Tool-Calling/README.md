# Aula 03: tool calling

**Origem:** o `run_tools` de [`openai_provider.py`](../../labs/llm_apis/openai_provider.py),
[`anthropic_provider.py`](../../labs/llm_apis/anthropic_provider.py) e
[`gemini_provider.py`](../../labs/llm_apis/gemini_provider.py), com `Tool`, `ToolRun` e
`run_tool` de [`types.py`](../../labs/llm_apis/types.py).

## O problema

Um modelo só produz texto. Ele não consulta um banco nem chama um serviço. Com *tool
calling*, a aplicação descreve as funções disponíveis, o modelo **pede** a execução e a
aplicação decide, executa e devolve o resultado.

```
aplicação ── pergunta + ferramentas ──> modelo
aplicação <── "chame consultar_pedido(P-1001)" ── modelo
aplicação executa a função
aplicação ── resultado ──> modelo
aplicação <── resposta final (ou novo pedido) ── modelo
```

| Passo | Arquivo | Conceito novo |
|---|---|---|
| 1 | [`01_pedido_de_ferramenta.py`](01_pedido_de_ferramenta.py) | Declarar a ferramenta e ler o pedido |
| 2 | [`02_devolver_resultado.py`](02_devolver_resultado.py) | Uma volta completa e o histórico |
| 3 | [`03_falhas_de_ferramenta.py`](03_falhas_de_ferramenta.py) | Falhas que voltam ao modelo |
| 4 | [`04_laco.py`](04_laco.py) | Laço, chamadas paralelas e limite |
| 5 | [`05_exemplo_do_lab.py`](05_exemplo_do_lab.py) | O código do laboratório |

Execute a partir da raiz do repositório:

```bash
uv run python aulas/03-Tool-Calling/01_pedido_de_ferramenta.py
```

O [`simulador.py`](simulador.py) funciona como o da
[aula 02](../02-Geracao-Streaming-e-Saida-Estruturada): os SDKs são reais e as respostas
são reproduções programadas. Nenhum modelo decide nada aqui: a "escolha" da ferramenta
está escrita no roteiro.

---

## Passo 1: o pedido

A ferramenta é descrita por **nome**, **descrição** e um **JSON Schema** dos argumentos.
O modelo usa a descrição para decidir quando chamar, por isso ela faz parte do prompt:
descrição vaga leva a chamada errada.

O schema sai do Pydantic (`model_json_schema()`) e é o mesmo para as três APIs:

| | OpenAI | Claude | Gemini |
|---|---|---|---|
| Campo do schema | `parameters` | `input_schema` | `parameters_json_schema` |
| Onde vem o pedido | itens `function_call` em `output` | blocos `tool_use` | partes `function_call` |
| Identificador | `call_id` | `id` | `id` |
| Argumentos | **texto** JSON | objeto | objeto |

- O adaptador da OpenAI usa `strict: False`, porque o schema do Pydantic nem sempre
  cumpre as regras do modo estrito. A validação fica por conta da aplicação.
- No Gemini, `automatic_function_calling` desligado impede o SDK de executar funções
  sozinho: o controle fica no seu código.

## Passo 2: devolver o resultado

O resultado precisa apontar para o pedido pelo identificador. A diferença maior está no
**histórico**:

| Provedor | Como a segunda requisição carrega a conversa |
|---|---|
| OpenAI | `previous_response_id` + apenas os `function_call_output`; a API guarda o resto |
| Claude | tudo reenviado: pergunta, blocos do assistente e uma mensagem de usuário com os `tool_result` |
| Gemini | tudo reenviado: `contents` com o turno do modelo e as `function_response` |

Pontos que quebram integrações:

- No Claude, os blocos do assistente voltam **intactos**, inclusive o de raciocínio
  (`thinking`) com a assinatura.
- No Gemini, as partes do modelo voltam com a **thought signature**. O SDK tem até um
  motivo de término para a falta dela (`MISSING_THOUGHT_SIGNATURE`).
- O `previous_response_id` depende de a OpenAI guardar a resposta. Se o armazenamento
  estiver desligado, a aplicação precisa reenviar o histórico.
- Guardado de um lado ou de outro, o histórico conta como token de entrada. Cada volta
  fica mais cara.

## Passo 3: falhas voltam ao modelo

Os argumentos são **entrada não confiável**: o modelo pode errar o nome de um campo, o
tipo ou inventar uma ferramenta. Em vez de exceção, a falha vira um `ToolRun` com
`is_error=True`.

```
válido           is_error=False enviado
campo ausente    is_error=True  argumentos inválidos: 1 validation error for OrderArgs
tipo errado      is_error=True  argumentos inválidos: 1 validation error for OrderArgs
função falha     is_error=True  KeyError: 'P-9999'
não existe       is_error=True  ferramenta desconhecida: apagar_tudo
argumento extra  is_error=False enviado
JSON cortado     is_error=True  argumentos inválidos: Unterminated string ...
```

- **Argumento extra passa em silêncio**: o Pydantic o ignora por padrão. Para recusar,
  use `model_config = ConfigDict(extra="forbid")`.
- O JSON cortado só existe na OpenAI, que manda os argumentos como texto. O laboratório
  deixava o `JSONDecodeError` derrubar o laço; isso foi corrigido antes desta aula.
- Outra correção: `Tool.run` usava `model_dump()`, que transforma modelos aninhados em
  dicionários. Agora usa `dict(args)`, e a função recebe as instâncias.

Cada API é avisada da falha de um jeito:

| Provedor | Sinal de erro |
|---|---|
| OpenAI | nenhum campo próprio: a mensagem vai em `output` |
| Claude | `is_error: true` no `tool_result` |
| Gemini | chave `error` no lugar de `result` em `response` |

No fim do passo, o modelo erra o nome do argumento, lê a mensagem e corrige. Por isso
vale devolver um erro **descritivo**.

## Passo 4: o laço

```python
for turn in range(1, max_turns + 1):
    message = ...  # 1. pergunta ao modelo
    usage += ...  # 2. soma o consumo
    if message.stop_reason != "tool_use":  # 3. terminou?
        return ...
    results = [...]  # 4. executa todos os pedidos
    messages = [...]  # 5. acrescenta ao histórico
raise ToolLoopError(...)
```

- **Chamadas paralelas:** um turno pode trazer vários pedidos. Todos os resultados vão
  juntos na mensagem seguinte. "Paralelas" descreve o pedido: o laboratório as executa
  uma após a outra.
- **Chamadas encadeadas:** o frete depende do resultado da consulta, então vem em outro
  turno.
- **`max_turns`** impede laço infinito e conta sem fim.
- O consumo é a soma de todas as requisições.

```
Com max_turns=2
  ToolLoopError: sem resposta final após 2 turnos
  requisições=2 execuções=3
```

Repare: a ferramenta do último turno **foi executada**, mas o modelo nunca recebeu o
resultado, e o erro não diz o que rodou. Se a ferramenta alterasse dados, a aplicação
não saberia o que ficou feito.

## Passo 5: o exemplo do laboratório

Os três adaptadores têm o mesmo `run_tools`, e `use(provider)` não sabe qual recebeu.
Compare as três versões lado a lado: a estrutura do laço é igual, e o que muda é como
ler o pedido, como montar o resultado e como carregar o histórico.

O laço é explícito de propósito. Na aula 11, o `create_agent` do LangChain assume esse
papel.

## Limitações que continuam no laboratório

- No `max_turns`, a última execução se perde (o exercício abaixo).
- No Claude, uma resposta cortada por limite de tokens no meio de um pedido de
  ferramenta devolve texto vazio como sucesso, sem executar nada.
- A mensagem de qualquer exceção vai ao modelo, e pode expor detalhes internos.
- O retorno é convertido com `str()`: um dicionário chega no formato do Python, não em
  JSON.
- Não há aprovação humana nem proteção contra executar duas vezes uma ação que altera
  dados. A aprovação humana aparece na aula 16.

## Onde o laboratório comprova isso

Em [`test_providers.py`](../../labs/llm_apis/tests/test_providers.py):

- `test_tool_loop_executes_tool_and_returns_result_to_model`
- `test_parallel_tool_calls_are_all_answered_in_one_turn`
- `test_tool_failures_are_reported_to_the_model`
- `test_openai_malformed_tool_arguments_are_reported_to_the_model`
- `test_tool_receives_nested_models_as_instances`
- `test_tool_loop_stops_at_max_turns`
- `test_openai_tool_loop_chains_previous_response`
- `test_gemini_tool_loop_preserves_thought_signature`

```bash
uv run pytest labs/llm_apis
```

## Perguntas de entrevista

Responda sem consultar o código.

1. Quem executa a ferramenta? Que consequências isso tem para a segurança?
2. Como o modelo decide qual ferramenta chamar? O que você faria se ele escolhesse
   errado com frequência?
3. Por que validar argumentos se a API recebeu o schema?
4. Quando uma falha deve voltar ao modelo e quando deve interromper a aplicação?
5. Como a conversa é mantida em cada uma das três APIs? Quais os custos de cada forma?
6. O que são chamadas paralelas? Quando você as executaria de fato em paralelo?
7. Como escolher o `max_turns`? O que fazer quando ele estoura?
8. Uma ferramenta emite um reembolso. Que proteções você colocaria?
9. O resultado de uma ferramenta traz um texto com instruções maliciosas. Qual o risco?

## Exercício

Resolva a perda da última execução no `max_turns`, escrevendo o teste antes. Compare
três caminhos:

1. `ToolLoopError` carrega `tool_runs` e `usage`.
2. No último turno, não executar as ferramentas.
3. No último turno, pedir a resposta final sem oferecer ferramentas.

Escolha um e explique o que acontece com uma ferramenta que altera dados.
