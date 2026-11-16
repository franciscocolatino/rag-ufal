"""Avaliação do pipeline: retrieval (hit@k, MRR) e respostas do LLM.

Métricas de retrieval — só nas perguntas que TÊM resposta nos documentos:
    hit@k   fração das perguntas em que algum trecho relevante aparece no top-k.
            hit@5 é a que mais importa: são os 5 trechos que vão para o LLM.
            Se o trecho certo não está lá, o LLM não tem como acertar.
    MRR@10  média de 1/posição do primeiro trecho relevante (0 se não está no
            top-10). Diferente do hit@k, premia ter o trecho certo LÁ EM CIMA:
            1º lugar vale 1, 2º vale 0,5, 5º vale 0,2.

Métricas de resposta (mais lentas: uma chamada ao LLM por pergunta):
    acurácia         resposta contém a resposta esperada (normalizada)
    recusa indevida  disse "não encontrei" numa pergunta que tem resposta
    citação correta  algum trecho citado ([Tn]) é da página esperada
    recusa correta   disse "não encontrei" numa pergunta SEM resposta — mede
                     se o sistema admite não saber em vez de alucinar

"Trecho relevante" = chunk de alguma página listada na evidência da pergunta.
Uma pergunta pode ter várias evidências porque o mesmo artigo aparece em dois
PDFs (o Estatuto/Regimento Geral consolidado inclui a Res. 114/2023 inteira).
"""

import json
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from rag_ufal import config
from rag_ufal.generate import answer, cited_results, is_no_answer
from rag_ufal.index import Index, SearchResult
from rag_ufal.ingest import Chunk

KS = (1, 3, 5, 10)


@dataclass
class Question:
    question: str
    answer: str | list[str] | None
    evidence: list[tuple[str, int | None]]  # (documento, página); página None = qualquer página
    type: str = ""

    @property
    def answerable(self) -> bool:
        return bool(self.evidence)


def load_questions(path: Path = config.EVAL_FILE) -> list[Question]:
    """Aceita `evidence: [{source, page}, ...]` ou o formato simples `source` + `page`
    (número, lista de páginas aceitas ou ausente)."""
    questions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if "evidence" in d:
                evidence = [(e["source"], e.get("page")) for e in d["evidence"]]
            elif d.get("source"):
                pages = d.get("page")
                pages = pages if isinstance(pages, list) else [pages]
                evidence = [(d["source"], p) for p in pages]
            else:
                evidence = []
            questions.append(Question(d["question"], d.get("answer"), evidence, d.get("type", "")))
    return questions


def is_relevant(chunk: Chunk, q: Question) -> bool:
    return any(chunk.source == src and (page is None or chunk.page == page) for src, page in q.evidence)


def first_relevant_rank(results: list[SearchResult], q: Question) -> int | None:
    """Posição (1-indexada) do primeiro trecho relevante, ou None se nenhum veio."""
    for rank, r in enumerate(results, start=1):
        if is_relevant(r.chunk, q):
            return rank
    return None


def retrieval_metrics(ranks: list[int | None]) -> dict[str, float]:
    metrics = {f"hit@{k}": mean(r is not None and r <= k for r in ranks) for k in KS}
    metrics["mrr@10"] = mean(1 / r if r is not None and r <= 10 else 0.0 for r in ranks)
    return metrics


def evaluate_retrieval(index: Index, questions: list[Question], modes: list[str]) -> dict:
    answerable = [q for q in questions if q.answerable]
    report = {}
    for mode in modes:
        ranks = [first_relevant_rank(index.search(q.question, max(KS), mode), q) for q in answerable]
        report[mode] = {
            "metrics": retrieval_metrics(ranks),
            "per_question": [{"question": q.question, "rank": r} for q, r in zip(answerable, ranks)],
        }
    return report


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", text).strip()


def answer_matches(expected: str | list[str], output: str) -> bool:
    """Critério simples e transparente: a resposta esperada aparece no texto gerado.

    Limitação conhecida: não reconhece paráfrases ("sete" vs "7,0"). Por isso o
    dataset aceita uma lista de formas válidas para a mesma resposta.
    """
    options = expected if isinstance(expected, list) else [expected]
    out = _normalize(output)
    return any(_normalize(e) in out for e in options)


def evaluate_answers(index: Index, questions: list[Question], mode: str, k: int) -> dict:
    rows = []
    for q in questions:
        results = index.search(q.question, k, mode)
        start = time.perf_counter()
        output = answer(q.question, results)
        latency = time.perf_counter() - start
        refused = is_no_answer(output)
        cited = [r for _, r in cited_results(output, results)]
        rows.append({
            "question": q.question,
            "answerable": q.answerable,
            "expected": q.answer,
            "output": output,
            "refused": refused,
            "correct": refused if not q.answerable else (not refused and answer_matches(q.answer, output)),
            "cited": bool(cited),
            "citation_correct": any(is_relevant(r.chunk, q) for r in cited),
            "retrieved_relevant": any(is_relevant(r.chunk, q) for r in results),
            "latency_s": round(latency, 2),
        })

    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    metrics = {
        "n_answerable": len(ans),
        "n_unanswerable": len(unans),
        "accuracy": mean(r["correct"] for r in ans) if ans else None,
        "false_refusal": mean(r["refused"] for r in ans) if ans else None,
        "citation_correct": mean(r["citation_correct"] for r in ans) if ans else None,
        "correct_refusal": mean(r["refused"] for r in unans) if unans else None,
        "mean_latency_s": mean(r["latency_s"] for r in rows),
    }
    return {"mode": mode, "k": k, "metrics": metrics, "per_question": rows}


def save_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **report}
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def retrieval_table(report: dict) -> str:
    cols = [f"hit@{k}" for k in KS] + ["mrr@10"]
    lines = ["| modo | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    for mode, data in report.items():
        lines.append(f"| {mode} | " + " | ".join(f"{data['metrics'][c]:.3f}" for c in cols) + " |")
    return "\n".join(lines)


def answers_table(report: dict) -> str:
    m = report["metrics"]
    pct = lambda v: "—" if v is None else f"{v:.0%}"  # noqa: E731
    return "\n".join([
        "| métrica | valor |",
        "|---|---:|",
        f"| acurácia (com resposta, n={m['n_answerable']}) | {pct(m['accuracy'])} |",
        f"| recusa indevida | {pct(m['false_refusal'])} |",
        f"| citação correta | {pct(m['citation_correct'])} |",
        f"| recusa correta (sem resposta, n={m['n_unanswerable']}) | {pct(m['correct_refusal'])} |",
        f"| latência média | {m['mean_latency_s']:.1f} s |",
    ])
