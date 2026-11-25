# Aula 07: pipeline RAG

**Origem:** [`labs/rag/pipeline.py`](../../labs/rag/pipeline.py)

## O problema

Um modelo não conhece os dados internos da loja e, quando não sabe, pode inventar. Em
**RAG** (geração aumentada por recuperação), a aplicação busca os trechos relevantes e
os entrega ao modelo junto com a pergunta.

```
ingestão:  documentos ──> trechos ──> embeddings ──> banco
consulta:  pergunta ──> embedding ──> busca ──> corte ──> prompt ──> modelo ──> citações
```

| Passo | Arquivo | Conceito novo |
|---|---|---|
| 1 | [`01_recuperar.py`](01_recuperar.py) | Recuperar os k mais próximos |
| 2 | [`02_prompt_com_contexto.py`](02_prompt_com_contexto.py) | Prompt fundamentado |
| 3 | [`03_citacoes_e_sem_resposta.py`](03_citacoes_e_sem_resposta.py) | Citações e recusa de responder |
| 4 | [`04_ingestao.py`](04_ingestao.py) | Ingestão segura |
| 5 | [`05_exemplo_do_lab.py`](05_exemplo_do_lab.py) | O código do laboratório |

```bash
uv run python aulas/07-Pipeline-RAG/01_recuperar.py
```

O modelo é o `FakeListChatModel` do LangChain: devolve respostas programadas.

---

## Passo 1: recuperar

A busca ordena por similaridade e devolve os `k` primeiros. Ela **sempre** devolve algo,
mesmo para uma pergunta fora do assunto. Por isso é preciso um corte.

- `k` pequeno pode deixar a resposta de fora; `k` grande gasta tokens e traz ruído.

## Passo 2: o prompt

- A instrução manda usar **somente** o contexto e citar as fontes por número.
- Cada trecho vai numerado e com a origem, o que permite conferir a citação.
- As marcas `<contexto>` separam dado de pergunta. Dizer que o contexto não é instrução
  **reduz**, mas não elimina, o risco de **injeção de prompt** por um documento.

## Passo 3: citações e ausência de resposta

- `min_score` descarta trechos fracos. Se nada sobra, o modelo **não é chamado**.
- A citação vem do modelo e é conferida em código: número fora do intervalo fica de fora.
- O laboratório só reconhecia `[1]`; um `[1, 2]`, formato comum, perdia as duas
  citações. Isso foi corrigido antes desta aula.
- Uma citação válida não prova que a frase está no trecho. Isso exige avaliar a
  **fidelidade** da resposta.

## Passo 4: ingestão

A ordem importa:

1. calcular os embeddings;
2. apagar os documentos que serão substituídos;
3. gravar.

Se o passo 1 falhar, nada foi perdido. Entre os passos 2 e 3 ainda há uma janela em que o
documento não existe.

## Passo 5: o exemplo do laboratório

- `RagPipeline` recebe banco, gerador de embeddings e modelo prontos. Os três são
  trocáveis porque são contratos.
- `ingest` usa `contextual_text` no embedding (aula 05).
- `retrieve` fica separado de `answer`, o que permite avaliar a busca sem modelo (aula 08).
- `Answer` devolve o texto, as citações e o que foi recuperado, para depurar.

## Limitações que continuam no laboratório

- Um documento que fica vazio, ou é apagado, permanece no banco: não há operação de
  remoção no pipeline (o exercício abaixo).
- Apagar e gravar não é atômico.
- O texto de um documento pode conter `</contexto>` e confundir a delimitação.
- `min_score` é um número fixo e depende do modelo de embeddings (aula 08).
- Cada pergunta é isolada: não há histórico de conversa nem reescrita da pergunta.
- Não há controle de acesso por usuário sobre os trechos.

## Onde o laboratório comprova isso

Em [`test_pipeline.py`](../../labs/rag/tests/test_pipeline.py):

- `test_answer_sends_context_and_maps_citations`
- `test_grouped_citations_are_mapped`
- `test_irrelevant_question_does_not_call_the_model`
- `test_category_filter_limits_retrieval`
- `test_shrunk_document_leaves_no_orphan_chunks`

```bash
uv run pytest labs/rag/tests/test_pipeline.py
```

## Perguntas de entrevista

1. Quando usar RAG em vez de ajuste fino ou de colocar tudo no contexto?
2. O que fazer quando a busca não acha nada?
3. Como saber se um erro veio da busca ou da geração?
4. Como reduzir alucinação? O que as citações garantem e o que não garantem?
5. Um documento da base contém instruções maliciosas. O que acontece?
6. Como impedir que um usuário veja trechos que não pode ver?
7. Como atualizar a base sem janela de indisponibilidade?
8. Como lidar com "e para o Pix?" numa conversa?

## Exercício

Acrescente ao pipeline uma operação para remover um documento, escrevendo o teste antes.
Depois decida o que `ingest` deve fazer quando um documento passa a ter zero trechos.
