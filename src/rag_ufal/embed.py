"""Wrapper fino sobre o SentenceTransformer.

Por que um wrapper? Dois detalhes que, se esquecidos, derrubam a qualidade do
retrieval sem dar erro nenhum:

1. Prefixos assimétricos. A família e5 foi treinada vendo "query: " antes de
   perguntas e "passage: " antes de documentos. Sem eles os vetores caem numa
   região do espaço que o modelo não conhece bem.
2. Normalização L2. Com vetores de norma 1, o produto escalar É a similaridade
   de cosseno — então a busca vira uma única multiplicação de matrizes.
"""

import logging

import numpy as np

from rag_ufal import config

log = logging.getLogger(__name__)


def _prefixes(model_name: str) -> tuple[str, str]:
    """(prefixo_query, prefixo_passage) conforme a família do modelo.

    Deixa a troca de modelo de embedding (um dos experimentos planejados)
    restrita a uma variável de ambiente.
    """
    name = model_name.lower()
    if "e5" in name:
        return "query: ", "passage: "
    return "", ""


class Embedder:
    def __init__(self, model_name: str = config.EMBED_MODEL, device: str = "cpu"):
        # Import tardio: carregar torch + transformers leva alguns segundos,
        # e comandos que só usam BM25 não precisam pagar esse custo.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.query_prefix, self.passage_prefix = _prefixes(model_name)

    @property
    def dim(self) -> int:
        return self.model.get_embedding_dimension()

    @property
    def max_tokens(self) -> int:
        """Textos mais longos que isso são truncados silenciosamente pelo modelo."""
        return self.model.max_seq_length

    def encode_passages(self, texts: list[str], batch_size: int = config.EMBED_BATCH_SIZE) -> np.ndarray:
        """Matriz (N, dim) float32 com linhas normalizadas."""
        texts = [self.passage_prefix + t for t in texts]
        self._warn_truncation(texts)
        return self._encode(texts, batch_size, progress=True)

    def _warn_truncation(self, texts: list[str]) -> None:
        """Avisa se algum texto passa do limite de tokens (o fim dele seria ignorado).

        Em português o e5 gera ~1,4 token por palavra, então chunks de 200
        palavras (~280 tokens) cabem com folga nos 512; chunks de 400+ palavras não.
        """
        lengths = [len(ids) for ids in self.model.tokenizer(texts)["input_ids"]]
        too_long = sum(n > self.max_tokens for n in lengths)
        if too_long:
            log.warning(
                "%d de %d chunks passam de %d tokens e serão truncados (maior: %d)",
                too_long, len(texts), self.max_tokens, max(lengths),
            )

    def encode_query(self, text: str) -> np.ndarray:
        """Vetor (dim,) float32 normalizado."""
        return self._encode([self.query_prefix + text], batch_size=1, progress=False)[0]

    def _encode(self, texts: list[str], batch_size: int, progress: bool) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=progress,
        )
        return vectors.astype(np.float32)
