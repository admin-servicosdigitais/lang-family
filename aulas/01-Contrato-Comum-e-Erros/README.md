# Aula 01: contrato comum e erros

**Origem:** [`labs/llm_apis/types.py`](../../labs/llm_apis/types.py) e
[`labs/llm_apis/base.py`](../../labs/llm_apis/base.py)

## O problema

OpenAI, Claude e Gemini fazem a mesma coisa com formatos de resposta, nomes de campos
e exceções diferentes. Se a aplicação usar cada SDK diretamente, trocar ou combinar
provedores obriga a mexer no código de negócio.

A solução do laboratório é uma camada com três peças: um **resultado** comum, um
**contrato** que todo provedor cumpre e **erros** normalizados. Com elas fica simples
fazer **fallback**.

```
código de negócio ──> LLMProvider (contrato) ──> adaptador ──> SDK ──> API
                  <── ChatResult / LLMError  <──
```

Nesta aula os adaptadores são falsos. Os reais são o tema da aula 02.

| Passo | Arquivo | Conceito novo |
|---|---|---|
| 1 | [`01_resultado.py`](01_resultado.py) | `dataclass` imutável e normalização da resposta |
| 2 | [`02_contrato.py`](02_contrato.py) | `Protocol` |
| 3 | [`03_erros.py`](03_erros.py) | Hierarquia de erros e tradução das exceções do SDK |
| 4 | [`04_fallback.py`](04_fallback.py) | Fallback entre provedores |
| 5 | [`05_exemplo_do_lab.py`](05_exemplo_do_lab.py) | Ferramentas e o contrato completo |

Execute a partir da raiz do repositório:

```bash
uv run python aulas/01-Contrato-Comum-e-Erros/01_resultado.py
```

---

## Passo 1: um resultado único

```python
@dataclass(frozen=True)
class ChatResult:
    text: str
    provider: str
    model: str
    usage: Usage = field(default_factory=Usage)
```

- `@dataclass` gera `__init__`, `__repr__` e `__eq__` a partir das anotações.
- `frozen=True` impede reatribuir campos. Um resultado é um fato que já aconteceu:
  ninguém deve alterá-lo no meio do caminho.
- `default_factory=Usage` chama `Usage()` para montar o valor padrão de cada instância.
  Esse recurso é obrigatório para padrões mutáveis, como listas e dicionários.
- `Usage.__add__` permite somar o consumo de várias chamadas com `+=`. Como o objeto é
  imutável, a soma cria um novo `Usage`.

As funções `from_openai` e `from_gemini` mostram o papel de um adaptador: conhecer o
formato de origem e devolver o tipo comum. No Gemini, os tokens de raciocínio vêm em
campo separado e precisam entrar na conta da saída.

```
openai: Pedido enviado. Usage(input_tokens=12, output_tokens=4)
gemini: Pedido enviado. Usage(input_tokens=11, output_tokens=23)
total: Usage(input_tokens=23, output_tokens=27)
ChatResult é imutável: a alteração foi bloqueada
```

## Passo 2: o contrato

```python
class LLMProvider(Protocol):
    name: str
    model: str

    def generate(self, prompt: str, *, system: str | None = None) -> ChatResult: ...
```

- `Protocol` define **tipagem estrutural**: uma classe cumpre o contrato por ter os
  atributos e métodos certos, sem herdar de nada. `FakeOpenAI` e `FakeAnthropic` não
  citam `LLMProvider`.
- Uma classe base abstrata (`ABC`) é **nominal**: exige herança e falha ao instanciar
  uma subclasse incompleta.
- O `*` obriga a passar `system` pelo nome, o que evita trocar os argumentos de lugar.
- `ask` recebe `LLMProvider`. É a inversão de dependência: o código de negócio depende
  do contrato, não de um SDK.

Quem confere o contrato é o checador de tipos. Sem o `# type: ignore`, o mypy responde:

```
error: Argument 1 to "ask" has incompatible type "ProviderWithoutGenerate"; expected "LLMProvider"  [arg-type]
note: "ProviderWithoutGenerate" is missing following "LLMProvider" protocol member:
note:     generate
```

Em execução, o Python não verifica nada: o erro só surge quando `generate` é chamado.

## Passo 3: erros normalizados

A pergunta que organiza a hierarquia é: **vale a pena tentar de novo?**

| Status | Erro | `retryable` | Motivo |
|---|---|---|---|
| 401, 403 | `AuthError` | não | A chave continuará errada |
| 429 | `RateLimitError` | sim | O limite se renova com o tempo |
| 408, 5xx, conexão | `UnavailableError` | sim | Falha temporária do serviço ou da rede |
| demais (400, 404...) | `BadRequestError` | não | A requisição continuará inválida |
| sem status | `RefusalError`, `ToolLoopError` | não | A API respondeu, mas o resultado não serve |

- `retryable` é um atributo de classe: as subclasses transitórias o sobrescrevem.
- `error_for_status` concentra o mapeamento em um só lugar para os três provedores.
- `_errors` é um *context manager* que traduz as exceções na fronteira com o SDK. Cada
  método do adaptador usa `with self._errors():` em vez de repetir o `try/except`.
- `raise ... from error` guarda a exceção original em `__cause__`. Normaliza-se o tipo
  sem perder a informação para depurar.

## Passo 4: fallback

```python
for provider in self.providers:
    try:
        return provider.generate(prompt, system=system)
    except LLMError as error:
        if not error.retryable:
            raise
        self.failures.append(error)
        last_error = error
```

- Só falhas transitórias passam para o próximo provedor.
- Erro de autenticação ou de requisição sobe na hora. Trocar de provedor esconderia um
  defeito de configuração ou de código.
- Se todos falharem, sobe o **último** erro.
- `failures` registra o que foi contornado, para observabilidade.

```
principal fora do ar: respondeu anthropic; chamadas ao reserva=1; falhas=['UnavailableError']
chave errada: erro AuthError de openai; chamadas ao reserva=0; falhas=[]
todos com falha: erro RateLimitError de anthropic; chamadas ao reserva=1; falhas=['UnavailableError', 'RateLimitError']
```

**Retry não é fallback.** Retry repete no mesmo provedor; fallback troca de provedor.
No laboratório o retry fica no SDK: os clientes da OpenAI e da Anthropic já tentam
duas vezes por padrão, e no Gemini é preciso ligar (visto na aula 02).

## Passo 5: o exemplo do laboratório

As definições dos blocos "do laboratório" são as mesmas de `types.py` e `base.py`. O que
aparece de novo são as ferramentas e o restante do contrato.

### Ferramentas

- `Tool` junta nome, descrição, modelo Pydantic dos argumentos e a função Python.
- `json_schema` sai do próprio modelo Pydantic: é o que o modelo de IA recebe para saber
  quais argumentos enviar.
- `Tool.run` valida os argumentos antes de executar. Os argumentos vêm de um modelo de
  IA e são **entrada não confiável**.
- Falhas viram um `ToolRun` com `is_error=True`, em vez de exceção. Assim o modelo
  recebe o problema e pode se corrigir.
- `run_tool` também trata o pedido de uma ferramenta que não existe.

São duas políticas de erro: falha da **API** vira exceção e sobe para a aplicação;
falha de **ferramenta** vira dado e volta ao modelo.

### Contrato completo

| Método | Devolve | Para quê |
|---|---|---|
| `generate` | `ChatResult` | Resposta inteira de uma vez |
| `stream` | `Iterator[str]` | Texto em pedaços, conforme é gerado |
| `structured` | instância de `T` | Resposta validada em um modelo Pydantic |
| `run_tools` | `ChatResult` | Laço de ferramentas até a resposta final |

`def structured[T: BaseModel](...) -> T` usa a sintaxe de genéricos do Python 3.12: quem
passa `schema=Ticket` recebe um `Ticket`, e o checador de tipos sabe disso.

O `FakeProvider` desta aula só devolve respostas fixas. Os métodos reais são o tema das
aulas 02 e 03.

## Onde o laboratório comprova isso

Em [`test_providers.py`](../../labs/llm_apis/tests/test_providers.py), com as APIs
simuladas na camada HTTP:

- `test_http_errors_are_normalized`
- `test_tool_failures_are_reported_to_the_model`
- `test_router_falls_back_to_next_provider_on_outage`
- `test_router_does_not_hide_configuration_errors`

```bash
uv run pytest labs/llm_apis
```

## Perguntas de entrevista

Responda sem consultar o código.

1. Por que criar uma camada própria sobre os SDKs? Quando isso não compensa?
2. Qual a diferença entre `Protocol` e `ABC`? Quando você escolheria cada um?
3. `frozen=True` torna o objeto totalmente imutável? Olhe para `ToolRun.arguments`.
4. Por que um 401 não aciona o fallback e um 503 aciona?
5. Qual a diferença entre retry e fallback? O que acontece com a latência quando os dois
   se somam?
6. O `FallbackRouter` cobre apenas `generate`. Por que o fallback em streaming é mais
   difícil?
7. Que problemas `failures` pode causar em um serviço que fica meses no ar?
8. Devolver a mensagem da exceção ao modelo tem algum risco?
9. O que muda para o usuário quando a resposta vem do provedor reserva?

## Exercício

No `04_fallback.py`, faça o roteador esperar um intervalo crescente antes de passar ao
próximo provedor quando o erro for `RateLimitError`. Receba a função de espera no
construtor, para conseguir testar sem esperar de verdade.
