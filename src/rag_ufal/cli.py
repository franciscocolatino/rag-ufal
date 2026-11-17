"""CLI: rag index | search | ask | eval."""

import argparse
import logging
import sys
import textwrap
import time
from pathlib import Path

import ollama

from rag_ufal import config
from rag_ufal.generate import cited_results, is_no_answer, stream_answer
from rag_ufal.index import SEARCH_MODES, Index, SearchResult


def cmd_index(args: argparse.Namespace) -> None:
    from rag_ufal.embed import Embedder
    from rag_ufal.ingest import load_corpus

    chunks = load_corpus(args.raw_dir, args.chunk_size, args.overlap)
    start = time.perf_counter()
    index = Index.build(chunks, Embedder(args.model), args.chunk_size, args.overlap)
    index.save(args.index_dir)
    print(
        f"{len(chunks)} chunks de {len(index.meta['sources'])} documentos "
        f"indexados em {time.perf_counter() - start:.0f}s -> {args.index_dir}"
    )


def cmd_search(args: argparse.Namespace) -> None:
    index = Index.load(args.index_dir)
    print_results(index.search(args.query, args.k, args.mode))


def cmd_ask(args: argparse.Namespace) -> None:
    index = Index.load(args.index_dir)
    if args.question:
        ask_once(index, args.question, args)
        return

    print("Modo interativo — Enter vazio ou Ctrl+D para sair.")
    while True:
        try:
            question = input("\npergunta> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        ask_once(index, question, args)


def ask_once(index: Index, question: str, args: argparse.Namespace) -> None:
    results = index.search(question, args.k, args.mode)
    if args.show_context:
        print_results(results)
        print()

    parts = []
    for token in stream_answer(question, results, args.llm):
        print(token, end="", flush=True)
        parts.append(token)
    print()

    text = "".join(parts)
    if is_no_answer(text):
        return
    cited = cited_results(text, results)
    if cited:
        print("\nFontes:")
        for n, r in cited:
            print(f"  [T{n}] {r.chunk.source}, p. {r.chunk.page}")
    else:
        print("\n(o modelo não citou nenhum trecho; trechos recuperados:)")
        for r in results:
            print(f"  - {r.chunk.source}, p. {r.chunk.page}")


def cmd_eval(args: argparse.Namespace) -> None:
    from rag_ufal import evaluate

    index = Index.load(args.index_dir)
    questions = evaluate.load_questions(args.questions)
    n_ans = sum(q.answerable for q in questions)
    print(f"{len(questions)} perguntas ({n_ans} com resposta, {len(questions) - n_ans} sem) | "
          f"índice: chunk {index.meta['chunk_size']}/{index.meta['chunk_overlap']}, {index.meta['embed_model']}\n")

    report = evaluate.evaluate_retrieval(index, questions, args.modes)
    evaluate.save_report({"index": index.meta, "retrieval": report}, args.out_dir / f"{args.tag}-retrieval.json")
    print("Retrieval:\n" + evaluate.retrieval_table(report))

    if args.answers:
        print(f"\nRespostas (modo {args.mode}, k={args.k}) — uma chamada ao LLM por pergunta...")
        report = evaluate.evaluate_answers(index, questions, args.mode, args.k)
        evaluate.save_report({"index": index.meta, "llm": config.LLM_MODEL, "answers": report},
                             args.out_dir / f"{args.tag}-answers.json")
        print(evaluate.answers_table(report))
    print(f"\nDetalhes por pergunta em {args.out_dir}/{args.tag}-*.json")


def print_results(results: list[SearchResult]) -> None:
    if not results:
        print("(nenhum resultado)")
    for rank, r in enumerate(results, start=1):
        print(f"{rank}. {r.chunk.source}, p. {r.chunk.page}  score={r.score:.4f}  ({r.chunk.id})")
        print(textwrap.indent(textwrap.shorten(r.chunk.text, width=300), "   "))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag", description="RAG sobre documentos da UFAL")
    parser.add_argument("--index-dir", type=Path, default=config.INDEX_DIR)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("index", help="lê os PDFs, gera embeddings e salva o índice")
    p.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    p.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    p.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    p.add_argument("--model", default=config.EMBED_MODEL, help="modelo de embedding")
    p.set_defaults(func=cmd_index)

    retrieval = argparse.ArgumentParser(add_help=False)
    retrieval.add_argument("-k", type=int, default=config.TOP_K, help="quantos trechos recuperar")
    retrieval.add_argument("--mode", choices=SEARCH_MODES, default=config.SEARCH_MODE)

    p = sub.add_parser("search", parents=[retrieval], help="só o retrieval (debug)")
    p.add_argument("query")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("ask", parents=[retrieval], help="pergunta com resposta do LLM")
    p.add_argument("question", nargs="?", help="omitir para o modo interativo")
    p.add_argument("--llm", default=config.LLM_MODEL, help="modelo no Ollama")
    p.add_argument("--show-context", action="store_true", help="mostra os trechos enviados ao LLM")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("eval", parents=[retrieval], help="métricas de retrieval (e, com --answers, das respostas)")
    p.add_argument("--questions", type=Path, default=config.EVAL_FILE)
    p.add_argument("--modes", nargs="+", choices=SEARCH_MODES, default=list(SEARCH_MODES),
                   help="modos de busca comparados no retrieval")
    p.add_argument("--answers", action="store_true", help="também avalia as respostas do LLM (lento)")
    p.add_argument("--tag", default="baseline", help="prefixo dos arquivos de resultado")
    p.add_argument("--out-dir", type=Path, default=config.RESULTS_DIR)
    p.set_defaults(func=cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    logging.getLogger("rag_ufal").setLevel(logging.INFO)
    logging.getLogger("pypdf").setLevel(logging.ERROR)  # avisos inofensivos em alguns PDFs

    try:
        args.func(args)
    except FileNotFoundError as e:
        sys.exit(f"erro: {e}")
    except ConnectionError:
        sys.exit("erro: não consegui falar com o Ollama — ele está rodando? (`ollama serve`)")
    except ollama.ResponseError as e:
        sys.exit(f"erro do Ollama: {e.error} — o modelo foi baixado? (`ollama pull {getattr(args, 'llm', config.LLM_MODEL)}`)")


if __name__ == "__main__":
    main()
