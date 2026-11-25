# Aula 17: Tree-of-Thoughts

**Origem:**
[`langgraph_patterns/tree_of_thoughts.py`](../../labs/langgraph_patterns/tree_of_thoughts.py),
baseado em Yao et al. (2023)

## O problema

Um modelo que raciocina em uma única linha não volta atrás quando erra cedo. No
**Tree-of-Thoughts**, ele propõe vários próximos passos, cada estado é avaliado e a busca
mantém só os mais promissores. O exemplo é o **Jogo do 24**: combinar quatro números com
as quatro operações até chegar a 24.

```
propor (um ramo por estado) ──> avaliar e podar ──> propor ──> ... ──> solução
```

| Passo | Arquivo | Conceito novo |
|---|---|---|
| 1 | [`01_passo_valido.py`](01_passo_valido.py) | Verificação em código |
| 2 | [`02_busca_em_largura.py`](02_busca_em_largura.py) | *Beam search* |
| 3 | [`03_send.py`](03_send.py) | `Send` |
| 4 | [`04_redutor_e_niveis.py`](04_redutor_e_niveis.py) | Redutor com reinício |
| 5 | [`05_exemplo_do_lab.py`](05_exemplo_do_lab.py) | O código do laboratório |

```bash
uv run python aulas/17-Tree-of-Thoughts/01_passo_valido.py
```

---

## Passo 1: o modelo propõe, o código confere

- O passo só vale se os números existirem e a conta for possível.
- `Fraction` dá aritmética exata: com ponto flutuante, `8 / (3 - 8/3)` não é 24.
- Notação como `1e999999999` é recusada antes do cálculo, que seria caríssimo.

## Passo 2: beam search

A cada nível expandem-se os estados mantidos, avaliam-se os filhos e ficam os `breadth`
melhores. Com largura 1 a busca é **gulosa**: um avaliador que se engana a leva a um beco
sem saída. Com largura 3, as alternativas permitem recuperar.

## Passo 3: `Send`

Cria um ramo por item, em quantidade definida na execução. Os ramos rodam no mesmo passo
do grafo, e o nó seguinte roda uma vez, depois de todos. Cada ramo recebe um estado
próprio (`input_schema`).

## Passo 4: redutor

Vários ramos escrevem na mesma chave, por isso ela precisa de um redutor que acumule. E
precisa esvaziar ao mudar de nível: devolver `None` limpa.

## Passo 5: o exemplo do laboratório

- `propose` roda em paralelo; `evaluate` junta, remove estados equivalentes, verifica a
  solução, avalia com `batch` e poda.
- Nota zero significa ramo impossível.
- O modelo não consegue declarar uma solução: ela é verificada nos números.

```
[2, 7, 8, 9] avaliador imperfeito, largura 1: sem solução (57 chamadas)
[2, 7, 8, 9] avaliador imperfeito, largura 3: resolvido  (86 chamadas)
```

Correções feitas no laboratório antes desta aula: uma proposta ou nota fora do formato
derrubava a busca inteira, e números em notação exponencial eram calculados.

## Quando não usar

O custo é alto: dezenas de chamadas para um único problema. Vale quando há busca real,
estados avaliáveis e solução verificável. Para a maioria das perguntas, uma chamada
resolve.

## Limitações que continuam no laboratório

- Um passo malformado descarta todas as propostas daquele estado (o exercício abaixo).
- As avaliações não têm limite próprio de concorrência.
- Não há orçamento total de chamadas.
- De estados equivalentes, guarda-se um só histórico.
- Uma entrada com um único número não é reconhecida como resolvida.

## Onde o laboratório comprova isso

Em
[`test_tree_of_thoughts.py`](../../labs/langgraph_patterns/tests/test_tree_of_thoughts.py):

- `test_finds_a_valid_solution`
- `test_wider_tree_recovers_from_a_misleading_evaluator`
- `test_frontier_is_expanded_in_parallel_branches`
- `test_model_cannot_claim_a_solution_it_did_not_compute`
- `test_unreadable_proposal_discards_only_that_branch`
- `test_unreadable_rating_prunes_the_state`
- `test_oversized_number_notation_is_rejected`

```bash
uv run pytest labs/langgraph_patterns/tests/test_tree_of_thoughts.py
```

## Perguntas de entrevista

1. Qual a diferença entre Chain-of-Thought e Tree-of-Thoughts?
2. O que a largura controla? Como escolher?
3. Por que verificar a solução em código?
4. Como funcionam `Send` e redutores no paralelismo?
5. Como estimar e limitar o custo?
6. O que fazer se o avaliador for ruim?

## Exercício

Faça um passo malformado descartar só ele, mantendo os demais da mesma proposta. Escreva
o teste antes e explique o que muda no schema pedido ao modelo.
