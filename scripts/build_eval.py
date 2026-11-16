"""Preenche a evidência (documento + página) de cada pergunta a partir da âncora.

Cada pergunta de data/eval/questions.jsonl tem um campo `anchor`: uma frase
LITERAL do PDF que contém a resposta. Este script procura a âncora em todas as
páginas de todos os PDFs e grava em `evidence` toda (documento, página) onde
ela aparece.

Por que assim, em vez de digitar a página à mão:
- elimina erro de digitação de página;
- acha duplicatas sozinho (o Estatuto/Regimento Geral consolidado contém a
  Res. 114/2023 inteira, então o mesmo artigo está em dois PDFs);
- a evidência é por PÁGINA, independente do chunking — dá para comparar
  tamanhos de chunk com o mesmo dataset.

Uso:  uv run python scripts/build_eval.py
"""

import json
import logging
import re
import sys
from pathlib import Path

from rag_ufal import config
from rag_ufal.ingest import read_pdf


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    pages = [
        (pdf.name, page, squash(text))
        for pdf in sorted(config.RAW_DIR.glob("*.pdf"))
        for page, text in read_pdf(pdf)
    ]

    path = config.EVAL_FILE
    questions = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    missing = []
    for q in questions:
        if not q.get("anchor"):
            q["evidence"] = []
            continue
        anchor = squash(q["anchor"])
        q["evidence"] = [{"source": src, "page": page} for src, page, text in pages if anchor in text]
        if not q["evidence"]:
            missing.append(q["question"])

    with open(path, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    n_multi = sum(len(q["evidence"]) > 1 for q in questions)
    print(f"{len(questions)} perguntas | {n_multi} com evidência em mais de uma página/documento")
    if missing:
        print("ÂNCORA NÃO ENCONTRADA:\n  " + "\n  ".join(missing))
        sys.exit(1)


if __name__ == "__main__":
    main()
