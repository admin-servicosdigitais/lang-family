# Aula 13: servidor MCP

**Origem:** [`mcp_lab/server.py`](../../labs/mcp_lab/server.py)

## O problema

Sem um padrão, cada aplicação de IA precisa de um conector próprio para cada sistema. O
**Model Context Protocol** padroniza essa ligação: um **servidor** expõe capacidades e
qualquer **cliente** compatível as usa.

```
aplicação (host) ── cliente MCP ══ protocolo (JSON-RPC) ══ servidor MCP ── sistemas
```

| Capacidade | O que é | Quem costuma decidir usar |
|---|---|---|
| Tool | ação que pode ser executada | o modelo |
| Resource | dado para leitura, identificado por URI | a aplicação |
| Prompt | roteiro pronto, com argumentos | o usuário |

| Passo | Arquivo | Conceito novo |
|---|---|---|
| 1 | [`01_primeira_ferramenta.py`](01_primeira_ferramenta.py) | `FastMCP` e uma ferramenta |
| 2 | [`02_validacao_e_saida_estruturada.py`](02_validacao_e_saida_estruturada.py) | Validação, saída e anotações |
| 3 | [`03_progresso_e_logs.py`](03_progresso_e_logs.py) | Notificações |
| 4 | [`04_resources_e_prompts.py`](04_resources_e_prompts.py) | Resources e prompts |
| 5 | [`05_exemplo_do_lab.py`](05_exemplo_do_lab.py) | O servidor do laboratório |

```bash
uv run python aulas/13-Servidor-MCP/01_primeira_ferramenta.py
```

---

## Passo 1: a primeira ferramenta

Nome, descrição e schema saem da função, como na aula 10. A sessão em memória permite
testar o protocolo sem processo nem rede.

## Passo 2: validação e saída estruturada

- Argumentos inválidos geram um resultado com `isError=True`, e o servidor continua no
  ar. O modelo pode ler e corrigir.
- Um retorno Pydantic vira `structuredContent`, com schema de saída declarado.
- `strip_whitespace` antes de `min_length` impede um motivo só de espaços.
- `readOnlyHint` e `destructiveHint` são **dicas**. O cliente não pode confiar nelas
  como controle de segurança.

## Passo 3: progresso e logs

O parâmetro `Context` é injetado pelo servidor e não aparece no schema. Com ele a
ferramenta informa progresso e envia logs durante a execução.

## Passo 4: resources e prompts

- `politicas://{name}` é um **modelo** de URI.
- A lista fechada impede acesso a arquivos: o nome nunca vira caminho.
- O argumento de um prompt entra no texto que vai ao modelo. Por isso ele é validado.

## Passo 5: o exemplo do laboratório

`create_server` cria uma instância com a própria lista de chamados, o que isola os
testes. `main` escolhe o transporte (aula 14).

```bash
uv run python -m labs.mcp_lab.server                               # stdio
uv run python -m labs.mcp_lab.server --transport streamable-http   # porta 8765
```

Correções feitas no laboratório antes desta aula:

- Um motivo só com espaços abria um chamado.
- O prompt aceitava qualquer texto como número de pedido.
- O número do pedido com espaços ou em minúsculas não era encontrado.

## Limitações que continuam no laboratório

- Repetir `abrir_chamado` cria chamados duplicados (o exercício abaixo).
- Não há autenticação nem autorização: quem se conecta consulta qualquer pedido.
- Argumentos extras são ignorados.
- Os chamados somem quando o processo termina.
- A mensagem das exceções chega ao cliente.

## Onde o laboratório comprova isso

Em [`test_server.py`](../../labs/mcp_lab/tests/test_server.py):

- `test_lists_tools_with_schemas_and_annotations`
- `test_failures_are_tool_errors_not_crashes`
- `test_progress_and_log_notifications_reach_the_client`
- `test_resource_template_only_serves_known_documents`
- `test_blank_reason_does_not_open_a_ticket`
- `test_prompt_rejects_text_that_is_not_an_order`

```bash
uv run pytest labs/mcp_lab/tests/test_server.py
```

## Perguntas de entrevista

1. Que problema o MCP resolve? Em que difere de uma API REST comum?
2. Qual a diferença entre tool, resource e prompt?
3. Que riscos um servidor com ferramentas de escrita traz?
4. Por que as anotações não bastam como segurança?
5. Como impedir acesso indevido a arquivos em um resource?
6. Como versionar as ferramentas sem quebrar clientes?

## Exercício

Torne `abrir_chamado` idempotente, escrevendo o teste antes. Decida o que identifica a
mesma solicitação: uma chave enviada pelo cliente ou o conteúdo.
