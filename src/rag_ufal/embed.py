"""Wrapper do SentenceTransformer: prefixos do e5 e vetores normalizados."""

import logging

import numpy as np

from rag_ufal import config

log = logging.getLogger(__name__)


def _prefixes(model_name: str) -> tuple[str, str]:
    """Modelos e5 foram treinados com "query: " e "passage: " antes do texto."""
    name = model_name.lower()
    if "e5" in name:
        return "query: ", "passage: "
    return "", ""


class Embedder:
    def __init__(self, model_name: str = config.EMBED_MODEL, device: str = "cpu"):
        # Import tardio: torch demora alguns segundos para carregar.
        from huggingface_hub.utils import logging as hub_logging
        from sentence_transformers import SentenceTransformer
        from transformers.utils import logging as transformers_logging

        hub_logging.set_verbosity_error()
        transformers_logging.set_verbosity_error()
        transformers_logging.disable_progress_bar()

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.query_prefix, self.passage_prefix = _prefixes(model_name)

    @property
    def dim(self) -> int:
        return self.model.get_embedding_dimension()

    @property
    def max_tokens(self) -> int:
        return self.model.max_seq_length

    def encode_passages(self, texts: list[str], batch_size: int = config.EMBED_BATCH_SIZE) -> np.ndarray:
        """Matriz (N, dim) float32 com linhas normalizadas."""
        texts = [self.passage_prefix + t for t in texts]
        self._warn_truncation(texts)
        return self._encode(texts, batch_size, progress=True)

    def _warn_truncation(self, texts: list[str]) -> None:
        """O modelo corta sem aviso o que passa de max_tokens; aqui avisamos."""
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
