# Aula 09: bancos vetoriais

**Origem:** [`labs/rag/stores/`](../../labs/rag/stores)

## O problema

Um banco vetorial guarda vetores com metadados e devolve os mais próximos de uma
consulta, com filtros. Cada produto tem API, nomes e escalas próprios. O laboratório
esconde isso atrás de um contrato e verifica todos com o mesmo teste.

| Passo | Arquivo | Conceito novo |
|---|---|---|
| 1 | [`01_busca_exata.py`](01_busca_exata.py) | Upsert, busca e filtro em memória |
| 2 | [`02_contrato.py`](02_contrato.py) | Contrato e teste de contrato |
| 3 | [`03_chroma_e_distancia.py`](03_chroma_e_distancia.py) | Banco real: distância x similaridade |
| 4 | [`04_tres_bancos.py`](04_tres_bancos.py) | Chroma, Qdrant e Milvus Lite |
| 5 | [`05_exemplo_do_lab.py`](05_exemplo_do_lab.py) | Os adaptadores do laboratório |

```bash
uv run python aulas/09-Bancos-Vetoriais/01_busca_exata.py
```

Os três bancos usados rodam embarcados, sem Docker e sem chave.

---

## Passo 1: o essencial

Comparar a consulta com todos os vetores é a **busca exata**: correta, mas com custo
proporcional ao tamanho da base. Bancos vetoriais usam índices de **vizinhos
aproximados** (ANN), como o HNSW, que trocam um pouco de recall por velocidade.

## Passo 2: o contrato

O teste usa vetores com cosseno conhecido e confere ordem, não duplicação e **escala** das
pontuações. A segunda implementação ordena certo, mas devolve distância: sem o teste, o
`min_score` do pipeline descartaria justamente os melhores trechos.

## Passo 3: distância x similaridade

O Chroma devolve distância de cosseno; o contrato pede similaridade, `1 - distância`. A
métrica precisa ser definida na criação da coleção, porque o padrão do Chroma não é
cosseno.

## Passo 4: diferenças entre os bancos

| Banco | Pontuação devolvida | Identificador | Filtro |
|---|---|---|---|
| Chroma | distância | texto | dicionário `where` |
| Qdrant | similaridade | inteiro ou UUID | objetos `Filter` |
| Milvus | similaridade, no campo `distance` | texto, com esquema | expressão em texto |
| Weaviate | distância | UUID | objetos `Filter` |
| Pinecone | similaridade | texto | dicionário com `$eq` |

- O **UUIDv5** gera sempre o mesmo UUID para o mesmo texto, o que mantém o upsert
  idempotente.
- No Milvus o filtro é uma expressão. Montá-la sem escapar permite **injeção**; o
  laboratório usa `json.dumps`.

## Passo 5: o exemplo do laboratório

- `payload_of` e `chunk_from` convertem o trecho para metadados e de volta.
- Cada adaptador usa o SDK nativo e entrega o mesmo comportamento.
- `close` libera conexões; `drop` apaga os dados.
- O Weaviate usa tokenização `FIELD` para o filtro ser exato. O Pinecone real tem
  **consistência eventual**: uma escrita demora a aparecer, e o teste espera por isso.

Um defeito foi corrigido no laboratório antes desta aula: gravar um lote vazio falhava no
Chroma, no Qdrant e no Weaviate, e era aceito nos outros.

## Como escolher

Pergunte: volume e taxa de escrita; quem opera (serviço gerenciado ou próprio); filtros e
multilocação; busca híbrida; custo; e o banco que a equipe já tem (o PostgreSQL com
`pgvector`, por exemplo, não faz parte deste laboratório).

## Limitações que continuam no laboratório

- Erros não são normalizados: dimensão errada ou `k=0` geram exceções diferentes em cada
  banco (o exercício abaixo).
- Uma coleção existente com outra dimensão não é conferida na abertura.
- No Milvus, o texto tem limite de 8192 caracteres definido no esquema.
- Weaviate e Qdrant em servidor exigem os contêineres de `infra/`.
- O Pinecone offline é um simulador próprio.

## Onde o laboratório comprova isso

Em [`test_store_contract.py`](../../labs/rag/tests/test_store_contract.py):

- `test_query_ranks_by_cosine_similarity`
- `test_upsert_with_same_id_replaces_record`
- `test_upsert_of_nothing_is_a_no_op`
- `test_metadata_filter`
- `test_filter_value_is_not_interpreted_as_expression`
- `test_delete_by_source`

```bash
docker compose -f infra/vectordbs.compose.yml up -d --wait   # opcional
uv run pytest labs/rag/tests/test_store_contract.py
```

## Perguntas de entrevista

1. O que é um banco vetorial e quando um banco comum basta?
2. Busca exata x aproximada: o que se perde e o que se ganha?
3. Cosseno, produto escalar e distância euclidiana: quando importam?
4. Filtrar antes ou depois da busca? Qual o problema de filtrar depois?
5. Como garantir ingestão idempotente?
6. Como separar os dados de clientes diferentes?
7. Como migrar de um banco vetorial para outro?

## Exercício

Faça dimensão errada e `k` menor que 1 gerarem o mesmo erro em todos os adaptadores,
escrevendo o teste de contrato antes.
