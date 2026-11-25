# Aula 06: embeddings

**Origem:** [`labs/rag/embeddings.py`](../../labs/rag/embeddings.py)

## O problema

Para buscar trechos por significado, cada texto vira um **vetor**. Textos parecidos
precisam ficar próximos. A busca então é: transformar a pergunta em vetor e achar os
vetores mais próximos.

| Passo | Arquivo | Conceito novo |
|---|---|---|
| 1 | [`01_vetores_e_cosseno.py`](01_vetores_e_cosseno.py) | Saco de palavras e cosseno |
| 2 | [`02_hashing.py`](02_hashing.py) | Dimensão fixa por hashing |
| 3 | [`03_normalizacao.py`](03_normalizacao.py) | Normalização |
| 4 | [`04_semantico.py`](04_semantico.py) | Modelo semântico e limite de tokens |
| 5 | [`05_exemplo_do_lab.py`](05_exemplo_do_lab.py) | O código do laboratório |

```bash
uv run python aulas/06-Embeddings/01_vetores_e_cosseno.py
```

Os passos 4 e 5 usam um modelo que roda na própria máquina, sem chave. Na primeira
execução são baixados cerca de 220 MB para `.cache/fastembed`.

---

## Passo 1: vetores e cosseno

No saco de palavras há uma posição por palavra do vocabulário. A **similaridade de
cosseno** mede o ângulo entre dois vetores e ignora o tamanho deles.

É uma busca **léxica**: "entrega" e "frete" não têm relação nenhuma para ela, e a segunda
pergunta zera contra todos os documentos.

## Passo 2: hashing

Em vez de montar um vocabulário, calcula-se a posição pelo hash da palavra. A dimensão
fica fixa e palavras novas não exigem nada.

- O preço são as **colisões**: palavras diferentes na mesma posição. Com mais dimensões,
  há menos colisões.
- Usa-se SHA-256 porque o `hash()` do Python muda a cada execução; os vetores gravados
  deixariam de valer.

## Passo 3: normalização

Dividir o vetor pelo próprio comprimento deixa o comprimento igual a 1. Daí:

- o produto escalar passa a ser o cosseno, mais barato de calcular;
- textos longos deixam de pontuar mais só por serem longos.

O vetor zero (texto vazio ou só pontuação) é mantido, para não dividir por zero.

## Passo 4: embeddings semânticos

O modelo foi treinado para aproximar textos de mesmo significado, inclusive em idiomas
diferentes.

- "A entrega é de graça?" encontra o frete grátis sem repetir as palavras.
- "How can I pay?" encontra o documento em português sobre pagamento.
- As pontuações não são porcentagens. Compare-as dentro do mesmo modelo; um corte como
  0,3 precisa ser calibrado para cada um.

### O limite que não avisa

Este modelo lê no máximo **128 tokens** e descarta o restante **em silêncio**: dois textos
longos com finais diferentes geram o mesmo vetor. Caracteres não são tokens: na base de
exemplo o maior trecho tem 49 tokens, mas 500 caracteres de códigos e valores passam de
250.

## Passo 5: o exemplo do laboratório

- `Embedder` é um `Protocol`: o pipeline não sabe qual gerador usa.
- `embed_documents` e `embed_query` são separados porque alguns modelos tratam
  documento e pergunta de formas diferentes. Neste, o caminho é o mesmo.
- `HashEmbedder` é determinístico e instantâneo: serve para testar a mecânica dos
  bancos vetoriais.
- `FastEmbedEmbedder` roda via ONNX, sem GPU e sem chave. O modelo é carregado só quando
  a classe é criada.
- `dimension` precisa coincidir com a configuração da coleção no banco.

```
HashEmbedder        64d  0.27  A garantia cobre defeitos de fabricação por um ano.
FastEmbedEmbedder  384d  0.55  Pedidos acima de R$ 299 têm frete grátis.
```

**Trocar de modelo exige reindexar tudo.** Vetores de modelos diferentes não são
comparáveis, mesmo com a mesma dimensão.

## Limitações que continuam no laboratório

- O corte em 128 tokens não gera erro nem aviso (o exercício abaixo).
- `HashEmbedder` com dimensão zero ou negativa falha com um erro pouco claro.
- A dimensão do `FastEmbedEmbedder` é uma constante e precisa acompanhar o modelo.

## Onde o laboratório comprova isso

Em [`test_retrieval_quality.py`](../../labs/rag/tests/test_retrieval_quality.py), que
compara os dois geradores na base de exemplo (aula 08):

```bash
uv run pytest labs/rag/tests/test_retrieval_quality.py
```

## Perguntas de entrevista

1. O que é um embedding?
2. Qual a diferença entre busca léxica e semântica? Quando a léxica é melhor?
3. Por que usar cosseno? O que muda com vetores normalizados?
4. Como escolher um modelo de embeddings para português?
5. Aumentar a dimensão sempre melhora? Qual o custo?
6. O que acontece se o texto passar do limite do modelo?
7. Como migrar para um novo modelo sem tirar o sistema do ar?
8. O que é busca híbrida e que problema ela resolve?

## Exercício

Faça o `FastEmbedEmbedder` detectar textos acima do limite, escrevendo o teste antes.
Compare três saídas: lançar erro, avisar no log ou dividir o texto e combinar os vetores.
Escolha uma e diga em que situação escolheria outra.
