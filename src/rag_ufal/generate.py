"""Geração: monta o prompt com os trechos recuperados e chama o LLM local (Ollama).

Decisão central — citação por marcador, não por nome:
o modelo vê os trechos marcados ([T1], [T2], ...) e cita só o marcador. Quem
traduz "[T2]" para "regimento-graduacao.pdf, p. 12" é o código. Um modelo de 3B
erra fácil ao copiar nome de arquivo e número de página; copiar "[T2]" é trivial.
O "T" evita confusão com números de artigo: com "[84]", o modelo citava o
Art. 84 do regimento em vez do trecho.
Assim a citação exibida nunca aponta para uma página inventada — no máximo
para o trecho errado, e isso a avaliação consegue medir.
"""

import re
from collections.abc import Iterator

import ollama

from rag_ufal import config
from rag_ufal.index import SearchResult

NO_ANSWER = "Não encontrei essa informação nos documentos."

# As escolhas abaixo (texto do prompt, ordem invertida, exemplo few-shot) vieram
# de uma comparação de variantes num mini-conjunto de perguntas — ver README.
# A versão original ("Se os trechos não contiverem a resposta, responda
# exatamente: ...") fazia o modelo de 3B recusar até com a resposta no trecho [1].
SYSTEM_PROMPT = f"""Você é um assistente que responde perguntas sobre documentos oficiais da Universidade Federal de Alagoas (UFAL).

Leia TODOS os trechos com atenção antes de responder. A resposta pode estar em qualquer um deles, inclusive no meio ou no fim de um trecho, e com palavras diferentes das da pergunta.

Regras:
1. Use somente as informações dos trechos.
2. Cite o marcador do trecho que sustenta a resposta (T1, T2, ...), entre colchetes. Exemplo: "O prazo é de 30 dias [T2]."
3. Somente se NENHUM trecho trouxer a informação, responda: "{NO_ANSWER}"
4. Responda em português, de forma direta e concisa."""

# Few-shot: um exemplo de resposta com citação e um de recusa. Modelos pequenos
# seguem um EXEMPLO de formato bem melhor do que uma regra escrita.
# O conteúdo é propositalmente absurdo (planeta fictício): num primeiro teste,
# com um exemplo sobre "biblioteca funciona de segunda a sexta", o modelo
# respondeu que o RU "funciona de segunda a sexta" — copiou o fato do exemplo.
_EXAMPLE_CONTEXT = (
    "Trechos:\n\n"
    "[T2] (exemplo.pdf, p. 7)\nA capital do Planeta Zorg é a cidade de Mirra.\n\n"
    "[T1] (exemplo.pdf, p. 3)\nO Planeta Zorg possui três luas, chamadas Abe, Bel e Cid.\n\n"
)
FEW_SHOT = [
    {"role": "user", "content": _EXAMPLE_CONTEXT + "Pergunta: Quantas luas tem o Planeta Zorg?"},
    {"role": "assistant", "content": "O Planeta Zorg tem três luas: Abe, Bel e Cid [T1]."},
    {"role": "user", "content": _EXAMPLE_CONTEXT + "Pergunta: Qual a população do Planeta Zorg?"},
    {"role": "assistant", "content": NO_ANSWER},
]


def format_context(results: list[SearchResult]) -> str:
    """Trechos marcados por relevância ([T1] = mais relevante), mas apresentados
    em ordem INVERSA: o mais relevante fica por último, colado na pergunta.

    Modelos pequenos dão mais peso ao texto próximo do ponto de geração; com o
    trecho certo no topo do prompt, o modelo o "esquecia" e recusava.
    """
    blocks = [
        f"[T{n}] ({r.chunk.source}, p. {r.chunk.page})\n{r.chunk.text}"
        for n, r in enumerate(results, start=1)
    ]
    return "\n\n".join(reversed(blocks))


def build_messages(question: str, results: list[SearchResult]) -> list[dict]:
    user = f"Trechos:\n\n{format_context(results)}\n\nPergunta: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *FEW_SHOT,
        {"role": "user", "content": user},
    ]


def stream_answer(question: str, results: list[SearchResult], model: str = config.LLM_MODEL) -> Iterator[str]:
    """Gera a resposta em pedaços (streaming), conforme o modelo produz os tokens."""
    if not results:
        # Sem nenhum trecho recuperado, não há o que o LLM possa usar de forma honesta.
        yield NO_ANSWER
        return

    response = ollama.chat(
        model=model,
        messages=build_messages(question, results),
        stream=True,
        options={
            # Temperatura baixa: queremos a resposta mais provável, não criatividade.
            "temperature": config.LLM_TEMPERATURE,
            # Janela de contexto em tokens (prompt + resposta). O default do Ollama
            # é menor e cortaria trechos; 4096 cabe na VRAM de 4 GB com o modelo 3B.
            "num_ctx": config.LLM_NUM_CTX,
        },
    )
    for part in response:
        yield part.message.content


def answer(question: str, results: list[SearchResult], model: str = config.LLM_MODEL) -> str:
    """Versão sem streaming (útil na avaliação)."""
    return "".join(stream_answer(question, results, model))


def cited_results(text: str, results: list[SearchResult]) -> list[tuple[int, SearchResult]]:
    """Trechos citados na resposta, na ordem da primeira citação: [(n, resultado), ...].

    Marcadores fora do intervalo (o modelo "inventou" um [T7] com 5 trechos) são ignorados.
    """
    cited = []
    for match in re.findall(r"\[T(\d+)\]", text):
        n = int(match)
        if 1 <= n <= len(results) and n not in (c for c, _ in cited):
            cited.append((n, results[n - 1]))
    return cited


def is_no_answer(text: str) -> bool:
    return "não encontrei" in text.lower()
