# Comparação de prompts (mini-conjunto de desenvolvimento)

Data: 15/11/2026 · LLM `qwen2.5:3b` (Ollama, temperatura 0.1, num_ctx 4096) ·
retrieval híbrido (RRF), k=5 · índice chunk 200/overlap 40, `multilingual-e5-small`.

**Atenção:** são só 8 perguntas, escritas para depurar o prompt e separadas do
dataset de avaliação oficial. Uma pergunta de diferença é ruído; isto serviu
para escolher a direção, não como resultado final.

## Perguntas (6 com resposta, 2 sem)

| Pergunta | Esperado |
|---|---|
| Qual a frequência mínima para ser aprovado em uma disciplina? | 75% |
| Qual a média parcial necessária para ser aprovado direto, sem prova final? | 7,0 |
| Que nota recebe o aluno que não participa de uma avaliação? | 0,0 (zero) |
| Quando começam as aulas do semestre 2026.2? | 27/07 |
| Quando começa o semestre letivo 2026.1? | 23/02 |
| Existe abono de faltas na UFAL? | não existe (Art. 83) |
| Qual o valor da bolsa de iniciação científica? | — recusar |
| Qual o horário de funcionamento do restaurante universitário? | — recusar |

## Resultados

| Variante | Corretas | Com citação |
|---|---:|---:|
| A — prompt inicial | 4/8 | 0/6 |
| B — A + ordem invertida (trecho mais relevante perto da pergunta) | 6/8 | 1/6 |
| C — prompt revisado (recusa só se NENHUM trecho responder) | 6/8 | 1/6 |
| D — C + ordem invertida | 6/8 | 3/6 |
| E — A + invertida + few-shot (exemplo sobre biblioteca) | 6/8 | 6/6 |
| F — D + few-shot (exemplo sobre biblioteca) | 6/8 | 6/6 |
| G — D + few-shot neutro (planeta fictício) | 7/8 | 5/6 |
| **G + marcador [Tn] (versão adotada)** | **7/8** | **5/6 válidas** |

## Observações

- **Prompt inicial recusava com a resposta no trecho [1]** (frequência de 75%):
  a regra "se não contiver, responda exatamente: …" era uma saída fácil demais
  para um modelo de 3B. Não era corte de contexto: o prompt tinha 1.883 tokens.
- **C e D alucinaram** sobre a bolsa de IC ("180 horas…").
- **Vazamento do few-shot (E, F):** com um exemplo dizendo "a biblioteca
  funciona de segunda a sexta", o modelo respondeu que o RU "funciona de
  segunda a sexta [2]". Exemplo neutro (planeta fictício) resolveu.
- **Número de artigo × número de trecho:** com marcadores `[1]..[5]`, o modelo
  citou `[84]` e `[76]` (números de artigos do regimento). Trocar para `[T1]..[T5]` resolveu.
- **Falha persistente:** "Existe abono de faltas?" — o Art. 83 ("Não existe o
  abono de falta") está no 2º trecho recuperado e o modelo ainda recusa;
  aparentemente confunde "a resposta é não" com "não há resposta".
