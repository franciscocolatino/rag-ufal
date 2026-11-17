"""Geração: monta o prompt com os trechos recuperados e chama o LLM local (Ollama).

O modelo cita só o marcador do trecho ([T1], [T2], ...); o código traduz o
marcador em documento e página. O "T" evita confusão com números de artigo.
"""

import re
from collections.abc import Iterator

import ollama

from rag_ufal import config
from rag_ufal.index import SearchResult

NO_ANSWER = "Não encontrei essa informação nos documentos."

# Prompt escolhido comparando variantes (results/prompt_ablation.md).
SYSTEM_PROMPT = f"""Você é um assistente que responde perguntas sobre documentos oficiais da Universidade Federal de Alagoas (UFAL).

Leia TODOS os trechos com atenção antes de responder. A resposta pode estar em qualquer um deles, inclusive no meio ou no fim de um trecho, e com palavras diferentes das da pergunta.

Regras:
1. Use somente as informações dos trechos.
2. Cite o marcador do trecho que sustenta a resposta (T1, T2, ...), entre colchetes. Exemplo: "O prazo é de 30 dias [T2]."
3. Somente se NENHUM trecho trouxer a informação, responda: "{NO_ANSWER}"
4. Responda em português, de forma direta e concisa."""

# Exemplos fictícios de propósito: com um exemplo realista, o modelo copiava fatos dele.
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
    """[T1] é o mais relevante, mas vai por último, perto da pergunta."""
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
    if not results:
        yield NO_ANSWER
        return

    response = ollama.chat(
        model=model,
        messages=build_messages(question, results),
        stream=True,
        options={"temperature": config.LLM_TEMPERATURE, "num_ctx": config.LLM_NUM_CTX},
    )
    for part in response:
        yield part.message.content


def answer(question: str, results: list[SearchResult], model: str = config.LLM_MODEL) -> str:
    return "".join(stream_answer(question, results, model))


def cited_results(text: str, results: list[SearchResult]) -> list[tuple[int, SearchResult]]:
    """[(n, resultado), ...] citados na resposta; ignora marcadores inexistentes."""
    cited = []
    for match in re.findall(r"\[T(\d+)\]", text):
        n = int(match)
        if 1 <= n <= len(results) and n not in (c for c, _ in cited):
            cited.append((n, results[n - 1]))
    return cited


def is_no_answer(text: str) -> bool:
    return "não encontrei" in text.lower()
