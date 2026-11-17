"""Command line for Local-RAG-DocQA.

    python ask.py 規程フォルダ "在宅勤務手当はいくらですか？"   one question
    python ask.py 規程フォルダ                                 interactive (empty line or Ctrl+C to quit)
    python ask.py 規程フォルダ "…" --json                      machine-readable answer
    python ask.py 規程フォルダ --endpoint http://127.0.0.1:11434/v1 --model qwen3:8b   (Ollama)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from rag_engine import SUPPORTED, LocalRagEngine


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ask.py", description="社内文書に、出典付きで答えます（外部送信なし）。")
    p.add_argument("docs", type=Path, help=f"文書フォルダまたはファイル（{' '.join(SUPPORTED)}）")
    p.add_argument("question", nargs="?", help="質問。省略すると対話モード")
    p.add_argument("--endpoint", default="http://127.0.0.1:8080/v1", help="OpenAI 互換のローカルサーバー（既定: llama-server）")
    p.add_argument("--model", default="qwen3.8-27b", help="モデル名（llama-server では無視される。Ollama では必須）")
    p.add_argument("--offline", action="store_true", help="LLM を使わず、該当段落をそのまま返す")
    p.add_argument("--json", action="store_true", help="回答を JSON で出力")
    return p


def show(answer, as_json: bool) -> None:
    if as_json:
        print(json.dumps(asdict(answer), ensure_ascii=False, indent=2))
        return
    print(answer.answer)
    if answer.grounded:
        print(f"  出典: {answer.source_title}  （{answer.mode}, スコア {answer.confidence}, {answer.latency_ms:.0f} ms）")
        first = answer.excerpt.split("\n", 1)[0]
        print(f"  該当: {first}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    engine = LocalRagEngine(endpoint=None if args.offline else args.endpoint, model_name=args.model)
    try:
        n = engine.load_directory(args.docs)
    except FileNotFoundError:
        print(f"見つかりません: {args.docs}", file=sys.stderr)
        return 2
    for s in engine.skipped:
        print(f"読み込めなかったファイル: {s}", file=sys.stderr)
    if n == 0:
        print(f"{args.docs} に読める文書がありません（対応: {' '.join(SUPPORTED)}）", file=sys.stderr)
        return 2
    if args.question:
        show(engine.query(args.question), args.json)
        return 0
    print(f"{n} 段落を読み込みました。質問をどうぞ（空行で終了）。", file=sys.stderr)
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        show(engine.query(q), args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
