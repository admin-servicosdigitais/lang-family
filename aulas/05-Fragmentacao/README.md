# Aula 05: fragmentação

**Origem:** [`labs/rag/chunking.py`](../../labs/rag/chunking.py)

## O problema

Em RAG, o modelo responde com base em trechos recuperados de uma base própria. Documentos
inteiros não servem como unidade: gastam contexto, e um único embedding para muitos
assuntos fica vago. Por isso o texto é dividido em **trechos** (*chunks*).

```
documento ──> [fragmentação] ──> trechos ──> embeddings ──> banco vetorial
                 aula 05                       aula 06         aula 09
```

Trecho pequeno demais perde contexto; grande demais dilui o assunto.

| Passo | Arquivo | Conceito novo |
|---|---|---|
| 1 | [`01_corte_fixo.py`](01_corte_fixo.py) | Tamanho fixo e sobreposição |
| 2 | [`02_splitter_recursivo.py`](02_splitter_recursivo.py) | Hierarquia de separadores |
| 3 | [`03_por_secao.py`](03_por_secao.py) | Divisão pela estrutura do documento |
| 4 | [`04_metadados_e_ids.py`](04_metadados_e_ids.py) | Metadados, ids estáveis e reingestão |
| 5 | [`05_exemplo_do_lab.py`](05_exemplo_do_lab.py) | O código do laboratório |

```bash
uv run python aulas/05-Fragmentacao/01_corte_fixo.py
```

---

## Passo 1: corte fixo

Cortar a cada N caracteres parte palavras e frases ao meio. A **sobreposição** repete o
fim de um trecho no começo do próximo, para uma ideia na fronteira aparecer inteira em
algum deles. O custo é mais trechos, mais armazenamento e conteúdo repetido nos
resultados.

## Passo 2: splitter recursivo

O `RecursiveCharacterTextSplitter` tenta separar por parágrafo, depois por linha, depois
por espaço e só então por caractere. Desce um nível apenas quando o pedaço ainda excede
`chunk_size`.

- `chunk_size` é um **máximo**, medido em caracteres por padrão, não em tokens.
- A sobreposição só acontece quando uma divisão precisa ser feita dentro de um bloco.

## Passo 3: por seção

O `MarkdownHeaderTextSplitter` divide nos títulos e guarda cada um nos metadados. Depois
o splitter recursivo limita o tamanho, mantendo os metadados.

Repare em dois comportamentos:

- Os títulos saem do texto do trecho e vão para os metadados.
- As linhas em branco desaparecem. A divisão por parágrafo deixa de ter o separador
  `\n\n` e passa a quebrar por linha.

## Passo 4: metadados, ids e reingestão

- **Front matter** é o cabeçalho entre `---`; dele sai a `category`, usada para filtrar
  buscas.
- **`contextual_text`** antepõe o título da seção ao texto no embedding. "Compras podem
  ser canceladas" fica mais fácil de achar quando se sabe que é sobre desistência.
- **Id estável** (`origem#posição`): reingerir o mesmo documento sobrescreve em vez de
  duplicar. É a **idempotência**.
- Só sobrescrever não basta. Se o documento encolhe, sobram **trechos órfãos** com
  informação antiga. A solução é apagar o documento e inserir de novo.

```
documento encolheu, só upsert : ['Só Pix agora.', 'Boleto aceito.']
substituindo o documento      : ['Só Pix agora.']
```

## Passo 5: o exemplo do laboratório

Junta tudo: front matter, divisão por seção e depois por tamanho (500 caracteres, 50 de
sobreposição), ids e `load_corpus`. Na base de exemplo são 14 trechos, o maior com 172
caracteres; nenhum precisa da divisão por tamanho.

Um defeito foi corrigido no laboratório antes desta aula: um `---` sem fechamento, como
uma régua horizontal no início do arquivo, derrubava a carga com `ValueError`. Quebras de
linha do Windows também faziam o cabeçalho virar conteúdo.

## Limitações que continuam no laboratório

- O título do documento não entra no `contextual_text`, só o da seção.
- O tamanho é medido em caracteres, sem relação direta com o limite do modelo de
  embeddings (ver aula 06).
- Um cabeçalho com fechamento esquecido vira conteúdo, sem aviso, e a categoria cai
  para `geral`.
- O id é posicional: inserir uma seção no começo muda os ids seguintes.
- Só Markdown. PDF, HTML e tabelas pedem outra extração.

## Onde o laboratório comprova isso

Em [`test_pipeline.py`](../../labs/rag/tests/test_pipeline.py):

- `test_chunking_keeps_section_and_category`
- `test_long_sections_are_split_within_size_limit`
- `test_reingesting_does_not_duplicate`
- `test_shrunk_document_leaves_no_orphan_chunks`
- `test_unclosed_front_matter_is_treated_as_content`
- `test_front_matter_with_windows_line_endings`

```bash
uv run pytest labs/rag/tests/test_pipeline.py
```

## Perguntas de entrevista

1. Por que fragmentar em vez de indexar o documento inteiro?
2. Como você escolheria o tamanho do trecho e a sobreposição? Como mediria se acertou?
3. Quando a divisão por estrutura é melhor que a por tamanho?
4. O que são metadados e que perguntas eles ajudam a responder?
5. O que acontece se um documento for atualizado e reingerido? E se for apagado?
6. Como você trataria um PDF com tabelas?
7. O que é contextualizar um trecho antes do embedding?

## Exercício

Inclua o título do documento no `contextual_text` e meça o efeito com
`uv run python -m labs.rag.evaluate` antes e depois. Escreva o teste antes e explique o
resultado, mesmo que não melhore.
