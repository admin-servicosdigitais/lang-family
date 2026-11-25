# Aulas

## Por que esta pasta existe

Os [laboratórios](../labs) são a evidência dos requisitos: código completo e testado.
Para estudar, porém, o código completo mostra o resultado e esconde o caminho até ele.

Esta pasta serve para **preparação para entrevistas**. Cada tema dos laboratórios vira
uma aula em cinco passos, do conceito mais básico até o código extraído do laboratório.

- `labs/` continua sendo a referência. As aulas copiam trechos dele para que cada passo
  rode sozinho; se um laboratório mudar, a aula correspondente precisa ser revista.
- Tudo é **simulado**: sem chaves de API, sem rede e sem custo.
- Por ser simulado, nada aqui comprova a integração com um serviço real.

## Estrutura de uma aula

```
aulas/NN-Nome-do-Tema/
├── README.md            explicação do código, perguntas de entrevista e exercício
├── 01_....py            conceito mais básico
├── 02_....py
├── 03_....py
├── 04_....py
└── 05_exemplo_do_lab.py o código como está no laboratório
```

Cada passo roda sozinho, com um comando. A partir do segundo, os comentários separam o
que já foi visto do que é novo. Uma aula pode ter um arquivo de apoio, como o
`simulador.py` da aula 02, para a simulação não esconder o conceito estudado.

Na aula 04, que trata de testes, os passos rodam com `uv run pytest <arquivo>`.

```bash
uv run python aulas/01-Contrato-Comum-e-Erros/01_resultado.py
```

## Como estudar

1. Leia a seção do passo no `README.md` da aula.
2. Leia o arquivo e tente prever a saída.
3. Execute e compare.
4. No fim, responda às perguntas de entrevista sem consultar o código.

## Ordem das aulas

A ordem vai do mais simples ao mais complexo, respeitando as dependências entre os
temas. O MCP vem antes do LangGraph porque o código dele é mais simples.

Situação: **aprovada**, **em validação** (escrita, aguardando aprovação) ou **pendente**
(não criada).

### Nível 1: chamar um modelo

| Aula | Tema | Origem em `labs/` | Situação |
|---|---|---|---|
| [01-Contrato-Comum-e-Erros](01-Contrato-Comum-e-Erros) | Resultado normalizado, `Protocol`, erros e fallback | `llm_apis/types.py`, `llm_apis/base.py` | aprovada |
| [02-Geracao-Streaming-e-Saida-Estruturada](02-Geracao-Streaming-e-Saida-Estruturada) | Os três SDKs: texto, streaming e Pydantic | `llm_apis/*_provider.py` | aprovada |
| [03-Tool-Calling](03-Tool-Calling) | Laço de ferramentas, limite de turnos e histórico | `run_tools` em `llm_apis/*_provider.py` | aprovada |
| [04-Testes-sem-Chave](04-Testes-sem-Chave) | Simulação na camada HTTP com o SDK real | `shared/fake_http.py`, `llm_apis/tests/` | aprovada |

### Nível 2: RAG

| Aula | Tema | Origem em `labs/` | Situação |
|---|---|---|---|
| [05-Fragmentacao](05-Fragmentacao) | Divisão por seção e tamanho, overlap, ids estáveis | `rag/chunking.py` | aprovada |
| [06-Embeddings](06-Embeddings) | Léxico x semântico, normalização, cosseno | `rag/embeddings.py` | aprovada |
| 07-Pipeline-RAG | Score mínimo, citações e reingestão | `rag/pipeline.py` | pendente |
| 08-Avaliacao-da-Recuperacao | recall@k e MRR | `rag/evaluate.py` | pendente |
| 09-Bancos-Vetoriais | Distância x similaridade, filtros, teste de contrato | `rag/stores/` | pendente |

### Nível 3: LangChain e observabilidade

| Aula | Tema | Origem em `labs/` | Situação |
|---|---|---|---|
| 10-Ferramentas-e-LCEL | `@tool`, composição de cadeias e roteamento | `observability/tools.py`, `observability/triage.py` | pendente |
| 11-Agente-LangChain | `create_agent`, limites e memória | `observability/agent.py`, `shared/fake_model.py` | pendente |
| 12-Tracing-e-PII | Callbacks, métricas e mascaramento | `observability/tracing.py` | pendente |

### Nível 4: MCP

| Aula | Tema | Origem em `labs/` | Situação |
|---|---|---|---|
| 13-Servidor-MCP | Tools, resources, prompts e validação | `mcp_lab/server.py` | pendente |
| 14-Transportes-e-Cliente-MCP | stdio, Streamable HTTP e agente cliente | `mcp_lab/agent.py` | pendente |

### Nível 5: padrões com LangGraph

| Aula | Tema | Origem em `labs/` | Situação |
|---|---|---|---|
| 15-Chain-of-Agents | Estado, nós, arestas condicionais e redutores | `langgraph_patterns/chain_of_agents.py` | pendente |
| 16-Supervisor | `Command`, `interrupt` e schema dinâmico | `langgraph_patterns/supervisor.py` | pendente |
| 17-Tree-of-Thoughts | `Send`, ramos paralelos e beam search | `langgraph_patterns/tree_of_thoughts.py` | pendente |
