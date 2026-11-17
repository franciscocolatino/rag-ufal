"""Caminhos, modelos e defaults. Qualquer valor pode ser sobrescrito por
variável de ambiente, ex.: RAG_CHUNK_SIZE=300 rag index
"""

import os
from pathlib import Path


def _env(name: str, default, cast=str):
    value = os.environ.get(f"RAG_{name}")
    return cast(value) if value is not None else default


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = _env("RAW_DIR", DATA_DIR / "raw", Path)
INDEX_DIR = _env("INDEX_DIR", DATA_DIR / "index", Path)
EVAL_FILE = _env("EVAL_FILE", DATA_DIR / "eval" / "questions.jsonl", Path)
RESULTS_DIR = _env("RESULTS_DIR", ROOT / "results", Path)

# Ingestão
CHUNK_SIZE = _env("CHUNK_SIZE", 200, int)  # em palavras
CHUNK_OVERLAP = _env("CHUNK_OVERLAP", 40, int)

# Embeddings
EMBED_MODEL = _env("EMBED_MODEL", "intfloat/multilingual-e5-small")
EMBED_BATCH_SIZE = _env("EMBED_BATCH_SIZE", 32, int)

# Busca
SEARCH_MODE = _env("SEARCH_MODE", "dense")
TOP_K = _env("TOP_K", 5, int)
RRF_K = _env("RRF_K", 60, int)

# Geração
LLM_MODEL = _env("LLM_MODEL", "qwen2.5:3b")
LLM_TEMPERATURE = _env("LLM_TEMPERATURE", 0.1, float)
LLM_NUM_CTX = _env("LLM_NUM_CTX", 4096, int)
