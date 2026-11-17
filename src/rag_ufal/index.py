"""Índice de busca: dense (NumPy), lexical (BM25) e híbrido (RRF).

Salva embeddings.npy, chunks.jsonl (mesma ordem da matriz) e meta.json.
O BM25 é reconstruído a partir dos chunks no load.
"""

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from rag_ufal import config
from rag_ufal.embed import Embedder
from rag_ufal.ingest import Chunk

SEARCH_MODES = ("dense", "bm25", "hybrid")


@dataclass
class SearchResult:
    chunk: Chunk
    score: float  # cosseno, BM25 ou RRF, conforme o modo


def tokenize(text: str) -> list[str]:
    """Tokens do BM25: minúsculas e sem acentos ("matrícula" == "matricula")."""
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.findall(r"\w+", text)


def _top_k(scores: np.ndarray, k: int) -> list[tuple[int, float]]:
    order = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in order]


class Index:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, meta: dict, embedder: Embedder | None = None):
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks mas {len(embeddings)} embeddings")
        self.chunks = chunks
        self.embeddings = embeddings
        self.meta = meta
        self._embedder = embedder
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    @property
    def embedder(self) -> Embedder:
        # A query usa o mesmo modelo que gerou o índice (meta.json), não o do config.
        if self._embedder is None:
            self._embedder = Embedder(self.meta["embed_model"])
        return self._embedder

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder, chunk_size: int, chunk_overlap: int) -> "Index":
        embeddings = embedder.encode_passages([c.text for c in chunks])
        meta = {
            "embed_model": embedder.model_name,
            "dim": embedder.dim,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "n_chunks": len(chunks),
            "sources": sorted({c.source for c in chunks}),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        return cls(chunks, embeddings, meta, embedder)

    def save(self, index_dir: Path = config.INDEX_DIR) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "embeddings.npy", self.embeddings)
        with open(index_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        (index_dir / "meta.json").write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, index_dir: Path = config.INDEX_DIR) -> "Index":
        if not (index_dir / "meta.json").exists():
            raise FileNotFoundError(f"índice não encontrado em {index_dir} — rode `rag index` primeiro")
        embeddings = np.load(index_dir / "embeddings.npy")
        with open(index_dir / "chunks.jsonl", encoding="utf-8") as f:
            chunks = [Chunk(**json.loads(line)) for line in f]
        meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        return cls(chunks, embeddings, meta)

    def search(self, query: str, k: int = config.TOP_K, mode: str = config.SEARCH_MODE) -> list[SearchResult]:
        if mode == "dense":
            hits = self._dense(query, k)
        elif mode == "bm25":
            hits = self._bm25(query, k)
        elif mode == "hybrid":
            hits = self._hybrid(query, k)
        else:
            raise ValueError(f"modo desconhecido: {mode!r} (use um de {SEARCH_MODES})")
        return [SearchResult(self.chunks[i], score) for i, score in hits]

    def _dense(self, query: str, k: int) -> list[tuple[int, float]]:
        # Vetores normalizados: produto escalar = cosseno.
        scores = self.embeddings @ self.embedder.encode_query(query)
        return _top_k(scores, k)

    def _bm25(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        return [(i, s) for i, s in _top_k(scores, k) if s > 0]

    def _hybrid(self, query: str, k: int, rrf_k: int = config.RRF_K, candidates: int = 50) -> list[tuple[int, float]]:
        """Reciprocal Rank Fusion: score = Σ 1 / (rrf_k + posição) nos dois rankings."""
        n = min(max(candidates, k), len(self.chunks))
        fused: dict[int, float] = {}
        for ranking in (self._dense(query, n), self._bm25(query, n)):
            for position, (i, _) in enumerate(ranking, start=1):
                fused[i] = fused.get(i, 0.0) + 1.0 / (rrf_k + position)
        return sorted(fused.items(), key=lambda item: item[1], reverse=True)[:k]
