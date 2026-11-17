"""Ingestão: PDF -> texto limpo por página -> chunks.

Um chunk nunca cruza páginas, para a citação de página ser exata.
"""

import logging
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

from pypdf import PdfReader

from rag_ufal import config

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Chunk:
    id: str  # ex.: "regimento-graduacao.pdf:p42:c1"
    source: str
    page: int  # 1-indexada
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # "regi-\nmento" -> "regimento"
    text = re.sub(r"[._…]{4,}", " ", text)  # pontilhado de sumário
    return re.sub(r"\s+", " ", text).strip()


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    """Janela deslizante de `size` palavras, com `overlap` palavras repetidas entre chunks."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError(f"precisa 0 <= overlap < size (recebi size={size}, overlap={overlap})")

    words = text.split()
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return chunks


def read_pdf(path: Path) -> list[tuple[int, str]]:
    """Retorna [(número_da_página, texto_limpo), ...], páginas 1-indexadas."""
    reader = PdfReader(path)
    return [(i, clean_text(page.extract_text() or "")) for i, page in enumerate(reader.pages, start=1)]


def chunk_pdf(path: Path, size: int, overlap: int) -> list[Chunk]:
    chunks = []
    empty_pages = 0
    for page_num, text in read_pdf(path):
        if not text:
            empty_pages += 1
            continue
        for i, piece in enumerate(chunk_words(text, size, overlap)):
            chunks.append(Chunk(id=f"{path.name}:p{page_num}:c{i}", source=path.name, page=page_num, text=piece))

    if empty_pages:
        log.warning("%s: %d página(s) sem texto extraível (imagem escaneada?)", path.name, empty_pages)
    return chunks


def load_corpus(
    raw_dir: Path = config.RAW_DIR,
    size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[Chunk]:
    """Lê todos os PDFs de `raw_dir`, em ordem alfabética."""
    pdfs = sorted(raw_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"nenhum PDF encontrado em {raw_dir}")

    chunks = []
    for pdf in pdfs:
        doc_chunks = chunk_pdf(pdf, size, overlap)
        log.info("%s: %d chunks", pdf.name, len(doc_chunks))
        chunks.extend(doc_chunks)
    return chunks
