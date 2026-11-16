# rag-ufal

Assistente que responde perguntas sobre documentos públicos da **Universidade
Federal de Alagoas** (Estatuto e Regimento Geral, regimento da graduação,
calendário acadêmico 2026, PPC de Ciência da Computação) em português, **citando
documento e página**, e dizendo quando a resposta não está nos documentos.

Roda 100% local e gratuito: embeddings na CPU, LLM de 3B no Ollama numa GPU de 4 GB.
O foco do projeto é **medir**: há um dataset de avaliação próprio, e cada
decisão do pipeline abaixo vem de um experimento com números.

```
$ rag ask "Na matrícula extraordinária dá para se matricular em disciplina que tem pré-requisito?"
Não é possível se matricular em disciplina que tem pré-requisito na matrícula extraordinária [T2].

Fontes:
  [T2] regimento-graduacao.pdf, p. 32

$ rag ask "Quanto custa a refeição no restaurante universitário?"
Não encontrei essa informação nos documentos.
```

## Resultados

Dataset próprio de 51 perguntas (43 com resposta, 8 sem). Configuração final:
busca dense, top-5, `qwen2.5:3b`.

- **Retrieval:** o trecho certo está entre os 5 primeiros em **86%** das perguntas
  (hit@5 = 0.860, MRR@10 = 0.715). Embeddings sozinhos venceram BM25 e o híbrido.
- **Respostas:** **63%** de acerto automático nas perguntas com resposta (67% após
  revisão manual dos erros) e **100% de recusa correta** nas perguntas sem resposta:
  o sistema não inventou nenhuma resposta. Latência ≈ 10 s por pergunta numa GTX 1650.
- **Ponto fraco:** a citação. Só 42% das respostas citam uma página que contém a resposta,
  e perguntas de calendário acertam 3/9.

### Retrieval: dense × BM25 × híbrido

43 perguntas com resposta, chunks de 200 palavras (overlap 40), `multilingual-e5-small`.

| modo | hit@1 | hit@3 | hit@5 | hit@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| **dense** (embeddings) | **0.628** | **0.791** | **0.860** | **0.930** | **0.715** |
| BM25 (lexical) | 0.302 | 0.535 | 0.628 | 0.721 | 0.448 |
| híbrido (RRF, k=60) | 0.419 | 0.605 | 0.767 | 0.884 | 0.556 |

- **hit@k**: fração das perguntas em que um trecho da página certa aparece entre os k primeiros.
  hit@5 é a que mais importa, porque são esses 5 trechos que vão para o LLM.
- **MRR@10**: média de 1/posição do primeiro trecho certo (1º lugar = 1, 2º = 0,5, …).

**O híbrido ficou pior que o dense sozinho.** O RRF dá peso igual às duas listas.
Com um BM25 fraco nestas perguntas, os resultados dele empurram para baixo os trechos
certos que o dense tinha posto no topo. Por isso a busca padrão do sistema é `dense`
(o híbrido continua disponível em `--mode hybrid`).
Ressalva: as perguntas foram escritas parafraseando os documentos (como um aluno
perguntaria), o que desfavorece a busca lexical.

### Tamanho do chunk

Mesmo dataset; evidência por página, então o chunking não altera o gabarito.

| chunk (palavras) | nº de chunks | chunks truncados (>512 tokens) | dense hit@5 | dense MRR@10 | BM25 hit@5 | híbrido hit@5 |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 1250 | 0 | 0.837 | 0.624 | 0.628 | 0.698 |
| **200** | 653 | 0 | **0.860** | **0.715** | 0.628 | 0.767 |
| 300 | 460 | 61 | 0.860 | 0.616 | 0.721 | 0.767 |
| 400 | 380 | 192 | 0.767 | 0.623 | 0.744 | 0.791 |

O dense piora com chunks grandes porque o e5 corta tudo que passa de 512 tokens:
no chunk de 400 palavras, metade dos chunks é truncada e o fim do texto nunca vira
vetor. O BM25 melhora com chunks maiores, já que mais palavras dão mais chance de casar
termos. Tabela completa em [`results/chunk_size.md`](results/chunk_size.md).

### Respostas do LLM

Mesmas 51 perguntas, top-5, `qwen2.5:3b` (temperatura 0.1). Uma chamada ao LLM por pergunta.

| métrica | dense | híbrido |
|---|---:|---:|
| acurácia (43 com resposta) | **63%** (27/43) | 58% (25/43) |
| recusa indevida (disse "não encontrei" tendo resposta) | 14% | 14% |
| citação correta (cita uma página com a resposta) | 42% | 49% |
| recusa correta (8 sem resposta) | **100%** | **100%** |
| latência média | 10,4 s | 11,6 s |

Por tipo de pergunta (dense): regras acadêmicas 10/16 · governança 6/8 · PPC 8/10 ·
**calendário 3/9** · sem resposta 8/8.

**De onde vêm os 16 erros (dense):**

- **5 são do retrieval:** a página certa não estava no top-5, e o LLM não tinha como acertar.
- **11 são do gerador:** o trecho certo estava no contexto e o modelo de 3B errou ou
  recusou. Exemplos: datas trocadas no calendário ("19/12" para o início das aulas de
  2026.2) e "02 anos" lido no lugar de "um (01) ano". **O gargalo atual é o LLM, não a
  busca**, e isso motiva o fine-tuning.
- **2 são falsos negativos do critério automático.** "4 (quatro) vezes" e "de 14 a 18 de
  dezembro" estão certos, mas não batem com as formas esperadas ("4 períodos",
  "14/12"). Com eles, a acurácia real fica em 29/43 = 67%.

**Citação e normas contraditórias.** Em 11 das 27 respostas certas, o trecho citado não é o
que contém a resposta. Um caso real: perguntado sobre o prazo da segunda chamada da prova
final, o modelo respondeu "2 dias úteis" (Res. 114/2023, vigente) mas citou a p. 38 do
Estatuto, a seção acadêmica antiga do Regimento Geral, que diz "48 horas". O
corpus mistura normas revogadas e vigentes, e o sistema ainda não sabe qual prevalece.

A diferença dense × híbrido nas respostas (2 perguntas na acurácia, 3 na citação) está
dentro do ruído de um dataset deste tamanho. A vantagem robusta do dense é no retrieval.
Detalhes por pergunta: [`results/dense-answers.json`](results/dense-answers.json).

### Prompt

O prompt foi ajustado num mini-conjunto separado de 8 perguntas
([`results/prompt_ablation.md`](results/prompt_ablation.md)): o prompt inicial
acertava 4/8 e não citava fontes; o final acerta 7/8. O que fez diferença:

- **A regra de recusa era uma saída fácil.** "Se os trechos não contiverem a resposta,
  responda exatamente: …" fazia o modelo de 3B recusar mesmo com a resposta no 1º trecho.
- **Ordem do contexto.** O trecho mais relevante vai por último, colado na pergunta.
- **Few-shot com exemplo neutro.** Um exemplo de resposta com citação fez o modelo citar.
  Mas um exemplo sobre "a biblioteca funciona de segunda a sexta" vazou: o modelo
  respondeu que o RU "funciona de segunda a sexta". O exemplo final é sobre um
  planeta fictício.
- **`[T1]` em vez de `[1]`.** Com números puros, o modelo citava `[84]`, que é o número do
  *artigo* 84 do regimento, não um trecho.

## Como funciona

```
INDEXAR   PDFs ─▶ texto por página ─▶ chunks de 200 palavras (overlap 40)
                                          │
                                          ├─▶ embeddings e5 ("passage: ...") ─▶ embeddings.npy
                                          └─▶ texto + documento + página ─────▶ chunks.jsonl

PERGUNTAR pergunta ─▶ "query: ..." ─▶ vetor ─▶ produto escalar com todos os chunks ─▶ top-5
          top-5 ─▶ prompt com trechos [T1..T5] ─▶ qwen2.5:3b (Ollama) ─▶ resposta com [Tn]
          [Tn] ─▶ (código) ─▶ "documento.pdf, p. 42"
```

Decisões principais:

- **Um chunk nunca cruza página:** a citação de página é sempre exata.
- **Busca vetorial em NumPy puro**, de propósito, sem banco vetorial. Com vetores
  normalizados, o cosseno vira um produto de matrizes: busca exata em milissegundos
  para centenas de chunks.
- **Prefixos `query:`/`passage:`** do e5, sem os quais o modelo perde qualidade sem dar erro.
- **Citação por marcador:** o LLM só copia `[T2]`; quem traduz para documento e página
  é o código. Um modelo de 3B erra ao copiar nomes de arquivo e números de página,
  então assim a citação exibida nunca aponta para uma página inventada.
- **Índice com `meta.json`**: o modelo de embedding da busca vem do índice, não da
  configuração atual. Codificar a pergunta com outro modelo daria vetores incomparáveis.

## Avaliação: o dataset

[`data/eval/questions.jsonl`](data/eval/questions.jsonl): 51 perguntas:

| tipo | nº | exemplo |
|---|---:|---|
| regras acadêmicas | 16 | "Perdi a prova final por estar doente. Até quando posso pedir segunda chamada?" |
| governança | 8 | "Quem monta a lista de candidatos a reitor enviada ao governo?" |
| PPC de Ciência da Computação | 10 | "Ciência da Computação é um curso noturno?" |
| calendário 2026 | 9 | "Qual o último dia para pedir trancamento de disciplina em 2026.2?" |
| **sem resposta nos documentos** | 8 | "Quanto custa a refeição no restaurante universitário?" |

- As perguntas foram rascunhadas com auxílio de um LLM a partir dos documentos e
  revisadas pelo autor.
- Cada pergunta traz uma **âncora**, uma frase literal do PDF que contém a resposta.
  [`scripts/build_eval.py`](scripts/build_eval.py) acha **todas** as páginas onde ela
  aparece. Isso importa porque o Estatuto consolidado inclui a Res. 114/2023 inteira:
  24 perguntas têm a resposta em mais de um documento, e o retrieval acerta se trouxer
  qualquer um deles.
- As perguntas sem resposta foram conferidas por busca no texto. Algumas são armadilhas
  de propósito: o RU aparece nos documentos, mas sem preço.
- Critério de acerto da resposta: a resposta esperada (ou uma forma equivalente listada)
  aparece no texto gerado. É simples e transparente, mas não reconhece paráfrases livres.

## Limitações conhecidas

- **Calendários são tabelas**, e o `pypdf` extrai tabela como texto corrido: datas
  podem se separar dos eventos. Perguntas de calendário dependem de o rótulo e a data
  ficarem próximos no texto extraído.
- **Cabeçalho repetido** ("UNIVERSIDADE FEDERAL DE ALAGOAS – UFAL SECRETARIA
  EXECUTIVA…") em toda página do Estatuto entra nos chunks como ruído.
- **Normas revogadas e vigentes no mesmo corpus.** O Regimento Geral tem regras
  acadêmicas antigas (ex.: segunda chamada em 48 horas) que a Res. 114/2023 substituiu
  (2 dias úteis). Não há metadado de vigência, então o retrieval pode trazer a versão antiga.
- **Perguntas sim/não com resposta negativa.** "A UFAL abona faltas?" foi recusada no
  conjunto de desenvolvimento, mesmo com "Não existe o abono de falta" no contexto.
  O modelo de 3B parece confundir "a resposta é não" com "não há resposta".
- **Dataset pequeno:** com 43 perguntas, cada uma vale 2,3 pontos percentuais. Diferenças
  de uma pergunta são ruído.

## Próximos passos

- **Fine-tuning (LoRA/QLoRA)** de um modelo de 0,5B–3B no Colab para responder no
  formato com citação e recusar corretamente, comparado com o modelo base nesta
  mesma avaliação.
- Publicar modelo e dataset no Hugging Face e uma demo em Gradio (HF Space).
- Retrieval: reranker (cross-encoder), RRF com pesos, remoção de cabeçalhos
  repetidos, extração de tabelas dos calendários.
- Avaliação de respostas com LLM-as-judge, além do critério por substring.

## Como rodar

Requisitos: Python 3.12, [`uv`](https://docs.astral.sh/uv/), [Ollama](https://ollama.com).

```bash
uv sync                        # dependências (torch CPU-only, ~1,2 GB)
ollama pull qwen2.5:3b         # LLM (1,9 GB)
uv run python scripts/download_pdfs.py  # baixa os 5 PDFs para data/raw/ (confere SHA-256)
uv run rag index               # lê PDFs, gera embeddings, salva em data/index/
uv run rag search "trancamento de matrícula"      # só o retrieval (debug)
uv run rag ask "Quando começa o semestre 2026.2?" # resposta com fontes
uv run rag ask                                    # modo interativo
uv run rag eval                # métricas de retrieval (dense/bm25/hybrid)
uv run rag eval --answers      # + métricas das respostas do LLM (~10 min)
```

Configuração por variável de ambiente (`RAG_CHUNK_SIZE`, `RAG_EMBED_MODEL`,
`RAG_SEARCH_MODE`, `RAG_LLM_MODEL`, …): ver [`config.py`](src/rag_ufal/config.py).

Documentos usados (o script acima os salva com estes nomes em `data/raw/`):

| arquivo | documento |
|---|---|
| `estatuto-regimento-geral.pdf` | [Estatuto e Regimento Geral da UFAL](https://ufal.br/transparencia/institucional/Estatuto_Regimento_Ufal.pdf) (consolidado) |
| `regimento-graduacao.pdf` | [Res. 114/2023-CONSUNI, atualizada pela Res. 122/2025](https://ufal.br/estudante/graduacao/coordenacoes/resolucoes-1/rco-n-114-de-05-12-2023.pdf) |
| `calendario-2026.pdf` | [Calendário Acadêmico 2026](https://ufal.br/estudante/graduacao/calendario-academico/2026/calendario-2026_final.pdf) |
| `calendario-2026-2.pdf` | [Calendário 2026.2 retificado (Res. 126/2026)](https://ufal.br/estudante/graduacao/calendario-academico/2026/calendario-2026-2_resolucao-no-126_2026-consuni-ufal.pdf) |
| `ppc-ciencia-computacao.pdf` | [PPC de Ciência da Computação 2019](https://ic.ufal.br/pt-br/graduacao/ciencia-da-computacao/documentos/projeto-pedagogico/projeto-pedagogico-2019/@@download/file) |

## Estrutura

```
src/rag_ufal/
  config.py     caminhos e defaults (sobrescrevíveis por env var)
  ingest.py     PDF -> texto limpo por página -> chunks
  embed.py      wrapper do SentenceTransformer (prefixos e5, normalização)
  index.py      busca dense (NumPy), BM25 e híbrida (RRF); save/load
  generate.py   prompt + Ollama, citação por marcador
  evaluate.py   hit@k, MRR, acurácia, recusa, citação
  cli.py        rag index | search | ask | eval
scripts/        download_pdfs.py, build_eval.py (gabarito por âncora), chunk_experiment.py
data/eval/      dataset de avaliação
results/        saídas dos experimentos
```

Stack: `pypdf`, `sentence-transformers` (`intfloat/multilingual-e5-small`),
`rank-bm25`, NumPy, Ollama (`qwen2.5:3b`).
