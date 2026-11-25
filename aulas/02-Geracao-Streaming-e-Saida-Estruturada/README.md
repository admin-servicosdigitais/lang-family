# Aula 02: geração, streaming e saída estruturada

**Origem:** [`openai_provider.py`](../../labs/llm_apis/openai_provider.py),
[`anthropic_provider.py`](../../labs/llm_apis/anthropic_provider.py) e
[`gemini_provider.py`](../../labs/llm_apis/gemini_provider.py). O `run_tools` fica para a
aula 03.

## O problema

A [aula 01](../01-Contrato-Comum-e-Erros) definiu o contrato com provedores falsos. Agora
entram os SDKs oficiais: cada um monta a requisição, entrega a resposta e falha do seu
jeito. O adaptador absorve essas diferenças.

| Passo | Arquivo | Conceito novo |
|---|---|---|
| 1 | [`01_geracao.py`](01_geracao.py) | A primeira chamada nos três SDKs |
| 2 | [`02_erros_e_retries.py`](02_erros_e_retries.py) | Exceções, recusa e novas tentativas |
| 3 | [`03_streaming.py`](03_streaming.py) | Resposta em pedaços |
| 4 | [`04_saida_estruturada.py`](04_saida_estruturada.py) | Resposta validada por Pydantic |
| 5 | [`05_exemplo_do_lab.py`](05_exemplo_do_lab.py) | Os três adaptadores do laboratório |

Execute a partir da raiz do repositório:

```bash
uv run python aulas/02-Geracao-Streaming-e-Saida-Estruturada/01_geracao.py
```

## Como a aula funciona sem chave

[`simulador.py`](simulador.py) é um arquivo de apoio. Cada SDK aceita um cliente HTTP, e
o simulador entrega um que responde com uma fila programada:

```python
sim = simulate(openai_text("Seu pedido foi enviado."))
client = openai.OpenAI(api_key="falsa", http_client=sim.http_client)
```

O SDK monta a requisição, interpreta a resposta e repete tentativas como faria de
verdade; `sim.requests` guarda o que foi enviado. Os formatos das respostas são
**reproduções**: o que se vê aqui não comprova o comportamento das APIs reais.

---

## Passo 1: a primeira chamada

| | OpenAI | Claude | Gemini |
|---|---|---|---|
| Método | `responses.create` | `messages.create` | `models.generate_content` |
| Pergunta | `input` | `messages` | `contents` |
| Instrução de sistema | `instructions` | `system` | `config.system_instruction` |
| Limite de saída | opcional | `max_tokens` obrigatório | opcional |
| Texto | `output_text` | blocos em `content` | `text` |
| Tokens | `usage.input_tokens` e `output_tokens` | iguais aos da OpenAI | `usage_metadata`, com raciocínio à parte |

- **Instrução de sistema** define comportamento; o prompt traz o pedido do usuário.
- **Tokens** são a unidade de cobrança e do limite de contexto. Entrada e saída costumam
  ter preços diferentes.
- No Claude a resposta é uma lista de blocos de vários tipos, por isso o filtro por
  `block.type == "text"`.
- No Gemini, somar `thoughts_token_count` à saída evita subestimar o custo.

A saída mostra o corpo JSON que cada SDK enviou. Compare onde cada um coloca a
instrução de sistema.

## Passo 2: erros, recusa e novas tentativas

### Exceções

| | OpenAI e Claude | Gemini |
|---|---|---|
| Resposta com status de erro | `APIStatusError` (`status_code`) | `errors.APIError` (`code`) |
| Falha de rede | `APIConnectionError` | `httpx2.TransportError` |

Um *context manager* por SDK traduz essas exceções para a hierarquia da aula 01.

### Recusa chega com status 200

Nenhuma exceção é lançada pelo SDK. É preciso olhar o conteúdo:

| Provedor | Sinal |
|---|---|
| OpenAI | um trecho da mensagem com `type == "refusal"` |
| Claude | `stop_reason == "refusal"` |
| Gemini | candidato sem conteúdo, com o motivo em `finish_reason` ou `prompt_feedback` |

### Novas tentativas

```
429 duas vezes e depois sucesso
  openai (padrão)            requisições=3 ok: Pedido enviado.
  anthropic (padrão)         requisições=3 ok: Pedido enviado.
  gemini (attempts=3)        requisições=3 ok: Pedido enviado.
  gemini (padrão)            requisições=1 RateLimitError (retryable=True)
```

- OpenAI e Claude repetem duas vezes por padrão (`max_retries=2`).
- O `google-genai` **não repete** por padrão: é preciso configurar
  `HttpRetryOptions(attempts=3)`. Atenção: `attempts` inclui a primeira chamada.
- O 401 gera uma única requisição nos três: o SDK sabe que repetir não adianta.
- Falhas de rede também são repetidas pelos SDKs que repetem.

Não escreva um laço de retry por cima do SDK sem desligar o dele: as tentativas se
multiplicam.

## Passo 3: streaming

O texto chega aos poucos. O tempo total é parecido, mas o primeiro pedaço aparece muito
antes, o que melhora a experiência em conversas.

| Provedor | Como consumir |
|---|---|
| OpenAI | `create(..., stream=True)` devolve eventos tipados; o texto vem em `response.output_text.delta` |
| Claude | `messages.stream(...)` oferece `text_stream` e `get_final_message()` |
| Gemini | `generate_content_stream(...)` devolve respostas parciais |

Três pontos que os entrevistadores costumam explorar:

1. **O gerador é preguiçoso.** Chamar `stream(...)` não envia nada. A requisição sai na
   primeira iteração, e é ali que o erro aparece. Um `try` em volta apenas da chamada
   não captura coisa alguma.

   ```
   openai     requisições antes/depois=0/1 [] LLMError
   ```

2. **A falha pode vir depois de texto entregue.** Não dá para desfazer o que o usuário
   já leu; quem consome precisa tratar a interrupção.

   ```
   anthropic  requisições antes/depois=0/1 ['Seu ', 'pedido '] RefusalError
   ```

3. **Recusa em streaming não é exceção do SDK.** Sem checagem explícita, o stream
   termina vazio e parece sucesso. O laboratório tinha esse defeito na OpenAI e no
   Gemini, corrigido antes desta aula:

   ```python
   elif event.type == "response.refusal.delta":
       raise RefusalError(self.name, "o modelo recusou a requisição")
   ```

   No Gemini, `_check_blocked` aceita só os motivos de término normais
   (`_NORMAL_FINISH`) e trata o restante como interrupção. Um motivo novo no SDK gera
   erro em vez de passar despercebido.

## Passo 4: saída estruturada

Em vez de pedir "responda em JSON" e torcer, envia-se o schema à API e valida-se a
resposta.

```python
class Ticket(BaseModel):
    category: str
    priority: int = Field(ge=1, le=5)
```

| Provedor | Como pedir | Campo no corpo | Resultado |
|---|---|---|---|
| OpenAI | `responses.parse(text_format=Ticket)` | `text.format` | `output_parsed` |
| Claude | `messages.parse(output_format=Ticket)` | `output_config.format` | `parsed_output` |
| Gemini | `response_mime_type` e `response_schema` | `generationConfig.responseSchema` | `parsed` |

Compare os schemas na saída:

- A OpenAI recebe `strict: true` e `additionalProperties: false`.
- O SDK do Claude move `minimum` e `maximum` para a descrição do campo. O limite vira
  orientação para o modelo, e a validação local é que o garante.
- O Gemini usa o próprio dialeto de schema, com tipos como `STRING` e `INTEGER`.

**Pedir um schema não dispensa validar.** A resposta ainda pode vir cortada, incompleta
ou fora de um limite. Cada SDK reage de um jeito:

| Provedor | Resposta inválida |
|---|---|
| OpenAI e Claude | `parse()` lança `ValidationError` do Pydantic |
| Gemini | não lança nada: `parsed` não é uma instância do modelo |

O adaptador converte os dois casos em `RefusalError`. Na OpenAI e no Claude, o
`ValidationError` vazava para quem chamava, quebrando o contrato; isso também foi
corrigido antes desta aula.

## Passo 5: o exemplo do laboratório

As três classes têm os mesmos métodos do laboratório, menos `run_tools`. Repare em:

- `__init__` recebe o cliente pronto (**injeção de dependência**). É o que permite
  testar sem rede.
- `_errors`, `_request`, `_usage` e `_result` concentram as diferenças de cada SDK.
- `model_for` lê o modelo de uma variável de ambiente, com um padrão no código.
- `use(provider)` não sabe qual provedor recebeu. É o contrato funcionando.

## Limitações que continuam no laboratório

- **Resposta cortada por limite de tokens** passa como sucesso em `generate` e `stream`
  nos três provedores.
- No Gemini, `generate` aceita um candidato bloqueado que ainda traga conteúdo: a
  checagem de `finish_reason` existe só no streaming.
- Uma falha informada por evento no stream da OpenAI vira `LLMError` genérico, sem
  nova tentativa, mesmo que seja temporária.
- Os adaptadores são síncronos.

## Onde o laboratório comprova isso

Em [`test_providers.py`](../../labs/llm_apis/tests/test_providers.py):

- `test_generate_sends_system_and_prompt_and_reads_usage`
- `test_rate_limit_is_retried_until_success` e `test_auth_error_is_not_retried`
- `test_stream_yields_chunks_in_order`
- `test_refusal_during_stream_becomes_a_typed_error`
- `test_gemini_blocked_prompt_during_stream_becomes_a_typed_error`
- `test_gemini_stream_finish_reason_after_partial_text`
- `test_structured_output_is_validated_into_model`
- `test_invalid_structured_output_becomes_a_typed_error`

```bash
uv run pytest labs/llm_apis
```

## Perguntas de entrevista

Responda sem consultar o código.

1. O que é um token e por que ele importa para custo e para o contexto?
2. Para que serve a instrução de sistema? Ela protege contra um usuário mal-intencionado?
3. Por que uma recusa não aparece como exceção do SDK?
4. Quais erros valem nova tentativa? Que cuidado tomar ao somar o seu retry ao do SDK?
5. Quando o streaming vale a pena, e quando atrapalha?
6. Onde fica o `try` para capturar o erro de um gerador? Por quê?
7. Como tratar uma falha depois que metade do texto já foi mostrada?
8. Qual a diferença entre pedir JSON no prompt e usar saída estruturada?
9. Se a API garante o schema, por que validar de novo?
10. O adaptador usa `RefusalError` para formato inválido. É um bom nome? O que você faria?

## Exercício

Faça os adaptadores detectarem a resposta cortada pelo limite de tokens, escrevendo o
teste antes. Os sinais são:

| Provedor | Sinal |
|---|---|
| OpenAI | `response.status == "incomplete"` |
| Claude | `stop_reason == "max_tokens"` |
| Gemini | `finish_reason == FinishReason.MAX_TOKENS` |

Decida antes: lançar um erro novo ou devolver o texto com uma marca no `ChatResult`?
Justifique a escolha.
