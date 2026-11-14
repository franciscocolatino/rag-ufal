"""Índice de busca: dense (embeddings + NumPy), lexical (BM25) e híbrido (RRF).

Arquivos salvos em INDEX_DIR:
    embeddings.npy  matriz (N, dim) float32, linhas normalizadas
    chunks.jsonl    um Chunk por linha, na MESMA ordem das linhas da matriz
    meta.json       modelo, parâmetros de chunking, etc. — para saber com que
                    configuração o índice foi gerado (essencial nos experimentos)

O BM25 não é salvo: ele é reconstruído a partir dos chunks no load. É rápido,
determinístico e evita serializar objetos Python com pickle.
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
    score: float  # escala depende do modo: cosseno, BM25 ou RRF — não comparar entre modos


def tokenize(text: str) -> list[str]:
    """Tokenização para o BM25: minúsculas, sem acentos, só sequências alfanuméricas.

    Remover acentos faz "matrícula" casar com "matricula" — comum em perguntas
    digitadas às pressas. NFD separa "í" em "i" + acento combinante, e
    descartamos os combinantes (categoria Unicode "Mn").
    """
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
        # A query PRECISA ser codificada pelo mesmo modelo que gerou o índice,
        # senão os vetores vivem em espaços diferentes e o cosseno não significa
        # nada. Por isso o modelo vem do meta.json, não do config atual.
        if self._embedder is None:
            self._embedder = Embedder(self.meta["embed_model"])
        return self._embedder

    # ---------- construção e persistência ----------

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

    # ---------- busca ----------

    def search(self, query: str, k: int = config.TOP_K, mode: str = "hybrid") -> list[SearchResult]:
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
        """Similaridade de cosseno = produto escalar, pois tudo está normalizado.

        (N, dim) @ (dim,) -> (N,): um score por chunk. Busca exata (força bruta);
        com alguns milhares de chunks isso leva milissegundos, então não há
        motivo para um índice aproximado (FAISS/HNSW) aqui.
        """
        scores = self.embeddings @ self.embedder.encode_query(query)
        return _top_k(scores, k)

    def _bm25(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        # Score 0 = nenhum termo da query aparece no chunk: não é um resultado.
        return [(i, s) for i, s in _top_k(scores, k) if s > 0]

    def _hybrid(self, query: str, k: int, rrf_k: int = config.RRF_K, candidates: int = 50) -> list[tuple[int, float]]:
        """Reciprocal Rank Fusion: score(d) = Σ_listas 1 / (rrf_k + posição_de_d).

        Usa só a POSIÇÃO em cada ranking, não o score bruto — por isso funciona
        mesmo com cosseno (~0.7–0.9) e BM25 (0 a dezenas) em escalas diferentes.
        rrf_k=60 (valor do artigo original) suaviza a vantagem do 1º lugar:
        1/61 vs 1/62 quase empatam, então aparecer bem nas DUAS listas vale
        mais que liderar só uma.
        """
        n = min(max(candidates, k), len(self.chunks))
        fused: dict[int, float] = {}
        for ranking in (self._dense(query, n), self._bm25(query, n)):
            for position, (i, _) in enumerate(ranking, start=1):
                fused[i] = fused.get(i, 0.0) + 1.0 / (rrf_k + position)
        return sorted(fused.items(), key=lambda item: item[1], reverse=True)[:k]
