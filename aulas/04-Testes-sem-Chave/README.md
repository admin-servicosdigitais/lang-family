# Aula 04: testes sem chave

**Origem:** [`llm_apis/tests/`](../../labs/llm_apis/tests),
[`shared/fake_http.py`](../../labs/shared/fake_http.py) e o
[`conftest.py`](../../conftest.py) da raiz.

## O problema

Testar chamando a API real custa dinheiro, é lento, exige chave e dá respostas que mudam.
Além disso, não dá para pedir à API que devolva um 429. A saída é substituir alguma peça,
e a pergunta de entrevista é: **em que ponto substituir?**

```
regra de negócio ──> adaptador ──> SDK ──> transporte HTTP ──> rede ──> API
                  ^ passo 1               ^ passos 2, 3, 5    ^ passo 4
```

Quanto mais à direita a substituição, mais código próprio e de terceiros é exercitado.

| Passo | Arquivo | Conceito novo |
|---|---|---|
| 1 | [`01_duble_no_contrato.py`](01_duble_no_contrato.py) | Dublê que cumpre o contrato |
| 2 | [`02_simular_o_transporte.py`](02_simular_o_transporte.py) | `MockTransport` no SDK oficial |
| 3 | [`03_cenarios_e_fixtures.py`](03_cenarios_e_fixtures.py) | Fixtures parametrizadas, retries e streaming |
| 4 | [`04_servidor_http_local.py`](04_servidor_http_local.py) | Servidor HTTP local |
| 5 | [`05_exemplo_do_lab.py`](05_exemplo_do_lab.py) | A estrutura do laboratório e o teste real |

Nesta aula os passos rodam com o pytest:

```bash
uv run pytest aulas/04-Testes-sem-Chave/01_duble_no_contrato.py -v
```

Não há `simulador.py`: o simulador é o assunto. A partir do passo 3, os adaptadores são
importados de `labs`, porque são o código sob teste.

---

## Passo 1: dublê no contrato

`FakeProvider` cumpre o `Protocol`, devolve o que o teste mandar e registra as chamadas.

- Rápido e simples, ideal para a **regra de negócio**: o que fazer em uma queda, que erro
  não esconder, o que é enviado como instrução de sistema.
- Não exercita o adaptador nem o SDK. Um campo escrito errado na requisição passaria.

Estruture cada teste em três blocos: preparar, executar e verificar.

## Passo 2: simular o transporte

```python
http_client = httpx2.Client(transport=httpx2.MockTransport(self.handle))
openai.OpenAI(api_key="teste", http_client=http_client, max_retries=0)
```

- O SDK serializa a requisição, interpreta a resposta e lança as próprias exceções.
- O teste confere **o que saiu** (caminho e corpo) e o que voltou.
- `max_retries=0` isola o comportamento que se quer observar.

O limite: o formato da resposta é uma **reprodução**. Se a API mudar, o teste continua
verde.

## Passo 3: cenários e fixtures

```python
@pytest.fixture(params=[FakeOpenAI, FakeAnthropic, FakeGemini])
def fake(request): ...
```

- Uma fixture parametrizada roda o mesmo teste uma vez por provedor: 4 testes viram 12.
  É um **teste de contrato**.
- Depois do `yield` vem a verificação final: nenhuma requisição inesperada e nenhuma
  resposta sobrando.
- O cabeçalho `retry-after-ms` deixa as novas tentativas quase instantâneas. No Gemini,
  os intervalos são configurados no cliente.
- Streaming é simulado com Server-Sent Events: linhas `event:` e `data:`.

## Passo 4: servidor HTTP local

Quando a biblioteca não aceita um transporte, mas aceita uma URL, sobe-se um servidor em
uma porta livre.

| | `MockTransport` | Servidor local |
|---|---|---|
| Precisa de | um ponto de injeção no cliente | configurar a URL base |
| Passa pela rede | não | sim, em `127.0.0.1` |
| Velocidade | maior | menor |
| Funciona com | bibliotecas sobre o mesmo cliente HTTP | qualquer biblioteca |

O laboratório usa o servidor local com o Pinecone e com os clientes de observabilidade.

## Passo 5: o exemplo do laboratório

- `FakeAPI` é uma classe base abstrata: cada provedor implementa como montar as respostas.
- `assert_all_used` e a fixture `autouse` `verify_fakes` conferem **todo** simulador, sem
  o teste precisar pedir.
- A marca `live` separa o teste real. O `conftest.py` da raiz o pula sem `--live`.

Duas proteções foram acrescentadas ao laboratório antes desta aula:

| Defeito | Proteção |
|---|---|
| O SDK do Claude transformava "requisição inesperada" em erro de conexão, e o teste podia passar pelo motivo errado | A requisição fica registrada e reprova o teste |
| Resposta programada e não usada passava em silêncio | Reprova o teste |
| Com um `.env` preenchido, um teste sem simulador chamaria a API paga | Sem `--live`, o `conftest.py` remove as credenciais do ambiente |

## O que cada nível prova

| Nível | Prova | Não prova |
|---|---|---|
| Dublê | a regra de negócio | adaptador e integração |
| HTTP simulado | adaptador, SDK e tratamento de erros | que o formato é o da API |
| Real (`--live`) | a integração naquele dia | cenários que não se consegue provocar |

Neste repositório as respostas são simuladas. Só diga que a integração real funciona
depois de rodar `uv run pytest --live`.

## Limitações que continuam no laboratório

- O simulador aceita qualquer caminho (o exercício abaixo).
- A rede não é bloqueada: a proteção é a ausência de chaves.
- No servidor local, uma exceção no `handle` aparece no cliente só como conexão
  encerrada.

## Perguntas de entrevista

1. Qual a diferença entre um dublê no contrato e a simulação do transporte?
2. Por que verificar a requisição enviada, além da resposta?
3. Como um teste pode passar pelo motivo errado? Dê um exemplo.
4. O que é uma fixture? E uma fixture `autouse`?
5. Como testar retries sem deixar a suíte lenta?
6. Os testes simulados passam e a produção quebra. Quais as causas prováveis?
7. Onde os testes reais entram no CI? Com que cuidados?
8. Como testar o comportamento de um modelo, cuja resposta varia?

## Exercício

Faça o simulador do laboratório recusar caminhos inesperados: cada subclasse de
`FakeAPI` declara o caminho que aceita, e uma requisição para outro endereço reprova o
teste. Escreva o teste antes.
