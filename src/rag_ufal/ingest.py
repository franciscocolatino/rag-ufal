"""Ingestão: PDF -> texto limpo por página -> chunks com (documento, página).

Decisão central: cada chunk pertence a UMA página. Isso torna a citação exata
("regimento-geral.pdf, p. 42") ao custo de cortar trechos que continuam na
página seguinte — o overlap entre chunks não atravessa páginas.
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
    id: str  # "regimento-geral.pdf:p42:c1" — estável e legível, útil no debug e na avaliação
    source: str  # nome do arquivo
    page: int  # 1-indexada, como o leitor de PDF mostra
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def clean_text(text: str) -> str:
    """Normaliza o texto extraído do PDF.

    - NFKC: desfaz ligaduras tipográficas ("ﬁ" -> "fi") e unifica acentos
      que o PDF pode codificar como letra + acento combinante.
    - Junta palavras hifenizadas na quebra de linha ("regi-\\nmento" -> "regimento").
    - Remove pontilhados de sumário ("Da Universidade........ 6"): viram
      "palavras" gigantes que estouram o limite de tokens do modelo de embedding.
    - Colapsa espaços/quebras de linha: o chunking é por palavras, então o
      layout não importa.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[._…]{4,}", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    """Janela deslizante de `size` palavras, andando `size - overlap` por passo.

    O overlap evita que uma informação fique cortada ao meio entre dois chunks
    sem aparecer inteira em nenhum deles.
    """
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError(f"precisa 0 <= overlap < size (recebi size={size}, overlap={overlap})")

    words = text.split()
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break  # a janela já chegou ao fim; outra seria só overlap repetido
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
        # Página sem texto extraível quase sempre é imagem escaneada (precisaria de OCR).
        log.warning("%s: %d página(s) sem texto extraível", path.name, empty_pages)
    return chunks


def load_corpus(
    raw_dir: Path = config.RAW_DIR,
    size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[Chunk]:
    """Lê todos os PDFs de `raw_dir` (em ordem alfabética, para o índice ser determinístico)."""
    pdfs = sorted(raw_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"nenhum PDF encontrado em {raw_dir}")

    chunks = []
    for pdf in pdfs:
        doc_chunks = chunk_pdf(pdf, size, overlap)
        log.info("%s: %d chunks", pdf.name, len(doc_chunks))
        chunks.extend(doc_chunks)
    return chunks
