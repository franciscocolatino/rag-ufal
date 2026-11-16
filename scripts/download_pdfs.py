"""Baixa os PDFs públicos da UFAL usados no projeto para data/raw/.

    uv run python scripts/download_pdfs.py

Cada arquivo é conferido pelo SHA-256 da versão usada na avaliação. Se a UFAL
publicar uma versão nova no mesmo link, o hash não bate: o script avisa (mas
mantém o arquivo), porque com outro texto os números do README podem mudar.
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# nome local -> (URL, sha256 da versão avaliada)
DOCUMENTS = {
    "estatuto-regimento-geral.pdf": (
        "https://ufal.br/transparencia/institucional/Estatuto_Regimento_Ufal.pdf",
        "75365cbd77083ac09c822532dd532f71a3f7760fcb477408ff570c3656883b78",
    ),
    "regimento-graduacao.pdf": (
        "https://ufal.br/estudante/graduacao/coordenacoes/resolucoes-1/rco-n-114-de-05-12-2023.pdf",
        "8c2524d922539dd7d49d39de844b2c2cccd3058eb84b7866034d6e1d83282504",
    ),
    "ppc-ciencia-computacao.pdf": (
        "https://ic.ufal.br/pt-br/graduacao/ciencia-da-computacao/documentos/projeto-pedagogico/"
        "projeto-pedagogico-2019/@@download/file",
        "bc6843cc488e8bf48dbe8ae4614d5e646f62b1c688c6d06df8f7ebc2c805a8fe",
    ),
    "calendario-2026.pdf": (
        "https://ufal.br/estudante/graduacao/calendario-academico/2026/calendario-2026_final.pdf",
        "51d6a9ba86addb70fb08d0e7ece4c5afddd7e1976270c82cf306bf8b2cc1437b",
    ),
    "calendario-2026-2.pdf": (
        "https://ufal.br/estudante/graduacao/calendario-academico/2026/"
        "calendario-2026-2_resolucao-no-126_2026-consuni-ufal.pdf",
        "0122764691c54d5fe2288f0fef2938a936d9c378a0924113e387929a1df622c7",
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    for name, (url, expected) in DOCUMENTS.items():
        path = RAW_DIR / name
        if path.exists() and sha256(path) == expected:
            print(f"já existe  {name}")
            continue
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "rag-ufal/0.1"})
            with urllib.request.urlopen(request, timeout=120) as response:
                path.write_bytes(response.read())
        except OSError as e:
            print(f"FALHOU     {name}: {e}")
            failed.append(name)
            continue
        if sha256(path) == expected:
            print(f"baixado    {name}")
        else:
            print(f"ATENÇÃO    {name}: versão diferente da avaliada — os resultados podem mudar")
    if failed:
        sys.exit(f"\n{len(failed)} download(s) falharam; baixe manualmente pelos links do README.")


if __name__ == "__main__":
    main()
