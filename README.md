# lang-family

Mini soluções em Python, cada uma com testes, para comprovar na prática os requisitos
técnicos de engenharia de IA: APIs de modelos, RAG com bancos vetoriais, LangChain com
observabilidade, padrões de LangGraph e MCP.

O domínio de exemplo é uma loja fictícia (Loja Aurora).

## Requisitos e onde estão as evidências

| Requisito | Laboratório | O que os testes comprovam |
|---|---|---|
| APIs OpenAI, Gemini e Claude | [`labs/llm_apis`](labs/llm_apis) | Mesmo contrato nos 3 SDKs oficiais: geração, streaming, saída estruturada, tool calling, retries, erros normalizados, fallback |
| RAG e bancos vetoriais | [`labs/rag`](labs/rag) | Chroma, Qdrant, Milvus, Weaviate e Pinecone atrás do mesmo contrato; ingestão idempotente, citações, avaliação com recall@k e MRR |
| LangChain com LangSmith/Langfuse | [`labs/observability`](labs/observability) | Agente com ferramentas e limites, cadeia LCEL, árvore de traces, tokens, sessão, score e mascaramento de CPF/e-mail |
| LangGraph e padrões de agentes | [`labs/langgraph_patterns`](labs/langgraph_patterns) | Supervisor-Driven Network (com aprovação humana), Tree-of-Thoughts e Chain-of-Agents |
| Model Context Protocol (MCP) | [`labs/mcp_lab`](labs/mcp_lab) | Servidor com tools, resources e prompts; stdio e Streamable HTTP; agente consumindo o servidor |
| Python | todo o repositório | Tipagem estrita (mypy), lint (ruff), pytest síncrono e assíncrono, `uv` com lockfile, CI |

## Como rodar

Requer [uv](https://docs.astral.sh/uv/) e, para parte dos testes, Docker.

```bash
uv sync
uv run pytest                    # offline: sem chaves e sem custo
uv run ruff check . && uv run mypy
```

Na primeira execução são baixados cerca de 220 MB do modelo local de embeddings.
Para pular esses testes: `uv run pytest -m "not embeddings"`.

### Três níveis de teste

| Nível | Comando | Precisa de |
|---|---|---|
| Offline | `uv run pytest` | nada além das dependências |
| Docker | suba os contêineres e rode `uv run pytest` | Docker; sem eles esses testes são pulados com o motivo |
| Real | `uv run pytest --live` | chaves no `.env` (veja `.env.example`); **gera custo** |

```bash
# Weaviate (127.0.0.1:18080) e Qdrant (127.0.0.1:16333)
docker compose -f infra/vectordbs.compose.yml up -d --wait

# Langfuse self-hosted: http://localhost:13000 (dev@example.com / senha-local-dev)
docker compose -f infra/langfuse.compose.yml up -d --wait

# Para parar (acrescente -v para apagar os dados)
docker compose -f infra/vectordbs.compose.yml down
docker compose -f infra/langfuse.compose.yml down
```

## O que cada nível prova (e o que não prova)

- Os testes **offline** provam a lógica própria: contratos, grafos, tratamento de erros,
  formato dos traces. Neles o LLM é um modelo roteirizado
  ([`labs/shared/fake_model.py`](labs/shared/fake_model.py)).
- As APIs da OpenAI, Anthropic e Gemini, o LangSmith e o Pinecone são **simulados na
  camada HTTP**, com os SDKs oficiais intactos. Os formatos simulados são reproduções.
- Só os testes **`--live`** provam a integração com esses serviços e o comportamento de
  um modelo real. Para usar este repositório como evidência, rode-os com suas chaves.
- Chroma, Milvus Lite e o Qdrant local rodam embarcados. Weaviate, Qdrant server e
  Langfuse são testados contra servidores reais em contêiner.

## Laboratórios

### 1. APIs de modelos — `labs/llm_apis`

Um protocolo `LLMProvider` com três adaptadores: OpenAI (Responses API), Claude
(Messages API) e Gemini (`google-genai`). O laço de ferramentas é explícito em cada um,
respeitando as particularidades: `previous_response_id` na OpenAI, blocos do assistente
devolvidos intactos no Claude e *thought signatures* preservadas no Gemini.

```bash
uv run pytest labs/llm_apis
```

### 2. RAG e bancos vetoriais — `labs/rag`

Documentos Markdown → fragmentação por seção → embeddings → banco vetorial → resposta
com citações. Sem trecho relevante, o modelo nem é chamado. Cada banco usa o SDK nativo
e passa pelo mesmo [teste de contrato](labs/rag/tests/test_store_contract.py), que
confere inclusive a escala dos scores (uns bancos devolvem distância, outros similaridade).

```bash
uv run pytest labs/rag
uv run python -m labs.rag.evaluate   # compara bancos e embeddings
```

Resultado medido (14 trechos, 11 perguntas, igual em Chroma, Qdrant e Milvus):

| Embeddings | recall@3 | MRR |
|---|---|---|
| Hash léxico (linha de base) | 0,55 | 0,29–0,33 |
| MiniLM multilíngue (semântico) | 1,00 | 0,91 |

É uma base pequena e fictícia: os números mostram o método de avaliação, não o
desempenho esperado em produção.

O emulador Pinecone Local foi descontinuado em 07/10/2026
([release notes](https://docs.pinecone.io/release-notes/2026.md)), por isso o teste
offline usa um [fake próprio](labs/rag/tests/fake_pinecone.py).

### 3. LangChain e observabilidade — `labs/observability`

Agente `create_agent` com ferramentas, limites de chamadas e memória por thread; cadeia
LCEL de triagem; rastreamento em LangSmith e Langfuse; métricas por callback.

```bash
uv run pytest labs/observability

# Ver um trace na interface, sem nenhuma chave (com o Langfuse local no ar):
LANGFUSE_PUBLIC_KEY=pk-lf-local-dev LANGFUSE_SECRET_KEY=sk-lf-local-dev \
LANGFUSE_BASE_URL=http://localhost:13000 uv run python -m labs.observability.demo
```

Atenção a quem for mascarar dados no Langfuse: o parâmetro `mask` não cobre a mensagem
de status dos erros. Aqui isso é tratado no `RedactingLangfuseHandler` e coberto por
testes. O mascaramento é por expressão regular (CPF e e-mail), não uma solução completa
de PII, e não alcança atributos como `user_id` e `session_id`.

### 4. Padrões de LangGraph — `labs/langgraph_patterns`

| Padrão | Implementação | Garantias testadas |
|---|---|---|
| [Supervisor-Driven Network](labs/langgraph_patterns/supervisor.py) | Supervisor roteia com saída estruturada e `Command` | Orçamento de passos, rota inválida, falha de trabalhador, `interrupt` para aprovação humana |
| [Tree-of-Thoughts](labs/langgraph_patterns/tree_of_thoughts.py) | Beam search no Jogo do 24, ramos paralelos com `Send` | Aritmética verificada em código; largura 3 resolve onde a busca gulosa falha |
| [Chain-of-Agents](labs/langgraph_patterns/chain_of_agents.py) | Trabalhadores em sequência + gerente | Cada agente vê só o seu trecho; unidade de comunicação com tamanho limitado |

```bash
uv run pytest labs/langgraph_patterns
```

### 5. MCP — `labs/mcp_lab`

Servidor `FastMCP` com saída estruturada, validação de argumentos, progresso, logs,
resources (com template protegido por lista fechada) e prompt.

```bash
uv run pytest labs/mcp_lab

# Conectar o servidor ao Claude Code:
claude mcp add loja-aurora -- uv run --directory "$PWD" python -m labs.mcp_lab.server

# Ou servir por HTTP em http://127.0.0.1:8765/mcp (sem autenticação: só uso local)
uv run python -m labs.mcp_lab.server --transport streamable-http
```
