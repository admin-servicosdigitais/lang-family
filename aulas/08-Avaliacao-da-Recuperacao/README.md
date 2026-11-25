# Aula 08: avaliação da recuperação

**Origem:** [`labs/rag/evaluate.py`](../../labs/rag/evaluate.py) e
[`golden.json`](../../labs/rag/data/golden.json)

## O problema

"Testei algumas perguntas e pareceu bom" não permite comparar duas versões. É preciso um
**conjunto de referência** (perguntas com o trecho que deveria ser recuperado) e métricas
calculadas sempre do mesmo jeito.

| Passo | Arquivo | Conceito novo |
|---|---|---|
| 1 | [`01_taxa_de_acerto.py`](01_taxa_de_acerto.py) | Conjunto de referência e recall@k |
| 2 | [`02_mrr.py`](02_mrr.py) | MRR |
| 3 | [`03_avaliar_um_recuperador.py`](03_avaliar_um_recuperador.py) | Função de avaliação e linha de base |
| 4 | [`04_limiar.py`](04_limiar.py) | Efeito do corte de pontuação |
| 5 | [`05_exemplo_do_lab.py`](05_exemplo_do_lab.py) | O código do laboratório |

```bash
uv run python aulas/08-Avaliacao-da-Recuperacao/01_taxa_de_acerto.py
```

Os passos 4 e 5 usam o modelo local de embeddings.

---

## Passo 1: recall@k

Com **um** trecho correto por pergunta, recall@k é a fração das perguntas em que ele
aparece entre os `k` primeiros (também chamado de *hit rate*). Com vários trechos
corretos, recall é a fração deles que foi recuperada.

## Passo 2: MRR

*Mean Reciprocal Rank* é a média de `1/posição` do primeiro acerto. Primeiro lugar vale
1, terceiro vale 0,33. Dois sistemas com o mesmo recall podem ter MRR bem diferentes, e
a posição importa porque o modelo e o orçamento de contexto favorecem o topo.

## Passo 3: comparar

A função recebe o recuperador, o que permite trocar a implementação e manter a régua.

- Sempre meça uma **linha de base**. Com 4 documentos e `k=3`, ignorar a pergunta já dá
  recall 0,75: um número só é bom em comparação com algo.
- Olhe os erros, não só a média.

## Passo 4: o corte de pontuação

```
corte  trecho certo entregue  fora do tema que passa
  0.2  11 de 11              1 de 3
 0.25  11 de 11              0 de 3
  0.3  10 de 11              0 de 3
```

Este é o ponto mais importante da aula. A avaliação do laboratório mede a busca **sem
corte** e dá recall@3 = 1,00. O pipeline usa `min_score = 0.3`, que descarta o trecho
certo de uma pergunta (rastreamento, com 0,29). **O número avaliado não é o
comportamento entregue.**

Aqui 0,25 entregaria tudo, mas escolher o corte olhando essas mesmas perguntas é ajuste
excessivo: a escolha precisa ser validada em perguntas separadas.

## Passo 5: o exemplo do laboratório

- `Case` identifica o acerto por arquivo e seção, não pelo id posicional.
- `evaluate` recebe banco, gerador e casos; sem casos, lança erro (antes desta aula
  falhava com divisão por zero).
- Semântico supera o léxico: recall@3 de 1,00 contra 0,55.

Para comparar os três bancos, como no laboratório:

```bash
uv run python -m labs.rag.evaluate
```

## Limitações que continuam no laboratório

- A avaliação ignora o `min_score` e usa `k=3`; o pipeline usa `k=4` (o exercício
  abaixo).
- São 11 perguntas sobre 14 trechos: servem para mostrar o método, não para prever o
  desempenho em produção.
- Não há perguntas sem resposta no conjunto de referência.
- Um erro de digitação em `golden.json` conta como falha da busca, sem aviso.
- Só a recuperação é avaliada. Fidelidade e qualidade da resposta ficam de fora.

## Onde o laboratório comprova isso

Em [`test_retrieval_quality.py`](../../labs/rag/tests/test_retrieval_quality.py) e
[`test_evaluate.py`](../../labs/rag/tests/test_evaluate.py):

- `test_semantic_retrieval_meets_quality_bar`
- `test_semantic_beats_lexical_baseline`
- `test_databases_agree_on_ranking`
- `test_evaluation_without_cases_is_rejected`

## Perguntas de entrevista

1. Como você montaria o conjunto de referência de um sistema sem dados rotulados?
2. Quando olhar recall e quando olhar MRR? Que outras métricas existem?
3. Como escolher `k` e o corte de pontuação?
4. A avaliação está ótima e os usuários reclamam. O que investigar?
5. Como avaliar a resposta gerada? Quais os riscos de usar um modelo como juiz?
6. Como colocar essa avaliação no CI sem torná-lo instável?

## Exercício

Faça a avaliação aceitar `min_score` e perguntas sem resposta, escrevendo o teste antes.
Relate acertos entregues e perguntas fora do tema que passaram, e proponha um corte
usando perguntas que não foram usadas para escolhê-lo.
