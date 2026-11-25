# Aula 14: transportes e cliente MCP

**Origem:** [`mcp_lab/agent.py`](../../labs/mcp_lab/agent.py) e a opção de transporte de
[`server.py`](../../labs/mcp_lab/server.py)

## O problema

O servidor da aula 13 precisa ser alcançado. O **transporte** define como as mensagens
trafegam, e o **cliente** descobre as ferramentas e as entrega a um agente.

| Passo | Arquivo | Conceito novo |
|---|---|---|
| 1 | [`01_ciclo_da_sessao.py`](01_ciclo_da_sessao.py) | Ciclo da sessão |
| 2 | [`02_stdio.py`](02_stdio.py) | Transporte stdio |
| 3 | [`03_streamable_http.py`](03_streamable_http.py) | Streamable HTTP |
| 4 | [`04_agente.py`](04_agente.py) | Agente com ferramentas MCP |
| 5 | [`05_exemplo_do_lab.py`](05_exemplo_do_lab.py) | O código do laboratório |

```bash
uv run python aulas/14-Transportes-e-Cliente-MCP/01_ciclo_da_sessao.py
```

Os passos 2 e 3 iniciam um processo servidor e o encerram ao final.

---

## Passo 1: a sessão

Inicializar (versão do protocolo e capacidades), descobrir (`list_tools`), usar
(`call_tool`) e encerrar. O cliente não precisa conhecer as ferramentas de antemão.

## Passos 2 e 3: os transportes

| | stdio | Streamable HTTP |
|---|---|---|
| Servidor | subprocesso do cliente | processo próprio, ouvindo em uma porta |
| Clientes | um | vários |
| Uso típico | ferramentas locais | serviço remoto ou compartilhado |
| Cuidados | nada além do protocolo na saída padrão | autenticação, TLS, validação de origem |

Escutando em `127.0.0.1`, o SDK recusa `Host` e `Origin` estranhos (421 e 403). Isso
protege contra **DNS rebinding**: um site tentando alcançar um servidor local pelo
navegador. **Não** substitui autenticação.

## Passo 4: o agente

`load_mcp_tools` converte as ferramentas do servidor em ferramentas do LangChain. Um
erro de ferramenta chega ao agente como mensagem, e o limite de chamadas impede o laço.

## Passo 5: o exemplo do laboratório

- `stdio_server_params` diz como iniciar o servidor, com o mesmo Python do cliente.
- `build_mcp_agent` recebe a sessão pronta: as ferramentas só funcionam enquanto ela
  estiver aberta.

Uma correção foi feita no laboratório antes desta aula: o agente não tinha limite
prático. Em uma sondagem, um modelo em laço fez 60 chamadas e só parou porque o roteiro
mandou; agora o padrão são 6.

## Limitações que continuam no laboratório

- Depois de fechar a sessão, chamar a ferramenta gera `ClosedResourceError`.
- Não há tempo limite próprio, reconexão nem autenticação.
- Descrições vindas do servidor entram no contexto do modelo. Um servidor malicioso pode
  usá-las para dar instruções (**envenenamento de ferramentas**).
- O agente recebe todas as ferramentas, sem filtro nem aprovação (o exercício abaixo).
- O `main` do laboratório exige um modelo real.

## Onde o laboratório comprova isso

Em
[`test_transports_and_agent.py`](../../labs/mcp_lab/tests/test_transports_and_agent.py):

- `test_streamable_http_transport_end_to_end`
- `test_agent_discovers_and_uses_mcp_tools`
- `test_looping_agent_stops_at_the_model_call_limit`

```bash
uv run pytest labs/mcp_lab/tests/test_transports_and_agent.py
```

## Perguntas de entrevista

1. Quando usar stdio e quando usar HTTP?
2. O que acontece na inicialização de uma sessão?
3. Como você autenticaria um servidor MCP remoto?
4. O que é DNS rebinding?
5. Que riscos há em conectar um agente a um servidor de terceiros?
6. Como escalar um servidor MCP que guarda estado por sessão?

## Exercício

Faça o agente receber só as ferramentas anotadas como somente leitura, a menos que o
chamador libere as demais. Escreva o teste antes e diga por que isso não basta contra um
servidor desonesto.
