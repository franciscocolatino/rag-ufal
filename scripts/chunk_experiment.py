"""Experimento: efeito do tamanho do chunk no retrieval (hit@k/MRR nos três modos).

Uso:  uv run python scripts/chunk_experiment.py
Saída: results/chunk_size.md e results/chunk_size.json
"""

import json
import logging

from rag_ufal import config, evaluate
from rag_ufal.embed import Embedder
from rag_ufal.index import SEARCH_MODES, Index
from rag_ufal.ingest import load_corpus

CONFIGS = [(100, 20), (200, 40), (300, 60), (400, 80)]  # overlap = 20% do chunk


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    logging.getLogger("pypdf").setLevel(logging.ERROR)

    embedder = Embedder()
    questions = evaluate.load_questions()
    rows = []
    for size, overlap in CONFIGS:
        index = Index.build(load_corpus(config.RAW_DIR, size, overlap), embedder, size, overlap)
        index.save(config.DATA_DIR / "index-exp" / f"cs{size}")
        report = evaluate.evaluate_retrieval(index, questions, list(SEARCH_MODES))
        for mode in SEARCH_MODES:
            rows.append({"chunk_size": size, "overlap": overlap, "n_chunks": len(index.chunks),
                         "mode": mode, **report[mode]["metrics"]})
        print(f"chunk {size}: {len(index.chunks)} chunks")

    cols = ["hit@1", "hit@3", "hit@5", "hit@10", "mrr@10"]
    lines = ["| chunk (palavras) | chunks | modo | " + " | ".join(cols) + " |",
             "|---:|---:|---|" + "---:|" * len(cols)]
    for r in rows:
        lines.append(f"| {r['chunk_size']}/{r['overlap']} | {r['n_chunks']} | {r['mode']} | "
                     + " | ".join(f"{r[c]:.3f}" for c in cols) + " |")
    table = "\n".join(lines)
    print(table)
    (config.RESULTS_DIR / "chunk_size.md").write_text(table + "\n", encoding="utf-8")
    (config.RESULTS_DIR / "chunk_size.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
