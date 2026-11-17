"""
Test suite for Local Privacy RAG Engine.
Validates retrieval, source attribution, and refusal on questions the documents do not answer,
both with the local LLM (if one is listening on the endpoint) and in offline extractive mode.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from rag_engine import LocalRagEngine

DOCS = Path(__file__).resolve().parent.parent / "samples" / "docs"

# question, expected source document (None = must refuse), fact that must appear in the excerpt
CASES = [
    ("返品は何日以内ですか？", "01_返品ポリシー", "14日"),
    ("予備機の貸出は何営業日前に申請しますか？", "02_予備機貸出規定", "2営業日"),
    ("東京23区の出張宿泊費の上限はいくらですか？", "03_出張旅費規程", "12,000円"),
    ("パスワードは何文字以上必要ですか？", "04_情報セキュリティ基本方針", "12文字"),
    ("グリーン車は使えますか？", "03_出張旅費規程", "役員以上"),
    ("オフィスのフリードリンクはどこに置いてありますか？", None, None),
]
# Not in the documents, but lexically close enough that only the LLM can tell.
LLM_ONLY_REFUSALS = ["有給休暇は何日ありますか？"]


def check(engine, question, want_doc, fact):
    a = engine.query(question)
    print(f"[{a.mode}] {question}\n  -> {a.answer}\n  source={a.source_title} score={a.confidence} {a.latency_ms}ms")
    if want_doc is None:
        assert not a.grounded and a.source_title == "—", f"must refuse: {question}"
        assert "確認できませんでした" in a.answer
    else:
        assert a.grounded, f"should be answered: {question}"
        assert want_doc in a.source_title, f"wrong source for {question}: {a.source_title}"
        assert fact in a.excerpt, f"excerpt for {question} lacks {fact}"
    return a


def run_tests():
    online = LocalRagEngine(endpoint="http://127.0.0.1:8080/v1")
    print(f"Indexed {online.load_directory(DOCS)} chunks from {DOCS}")
    probe = online.query(CASES[0][0])
    if probe.mode == "llm":
        print("\n=== With local LLM ===")
        for q, doc, fact in CASES:
            check(online, q, doc, fact)
        for q in LLM_ONLY_REFUSALS:
            check(online, q, None, None)
    else:
        print("\n(no LLM on 127.0.0.1:8080, skipping the LLM pass)")

    print("\n=== Offline extractive mode ===")
    offline = LocalRagEngine(endpoint=None)  # never calls an LLM
    offline.load_directory(DOCS)
    for q, doc, fact in CASES:
        a = check(offline, q, doc, fact)
        assert a.mode in ("extractive", "refused")

    run_format_tests()
    run_cli_tests()
    print("\n>>> ALL RAG TESTS PASSED <<<")


def run_format_tests():
    """Word is read with the standard library; PDF needs pypdf and is skipped (and reported) without it."""
    print("\n=== Word / PDF ===")
    eng = LocalRagEngine(endpoint=None)
    eng.load_directory(DOCS.parent / "formats")
    a = check(eng, "育児休業の申出はいつまでに行いますか？", "05_育児休業規程", "1か月前")
    assert a.page is None
    try:
        import pypdf  # noqa: F401
    except ImportError:
        assert any("pypdf" in s for s in eng.skipped), eng.skipped
        print("(pypdf not installed: the PDF was skipped and reported)")
        return
    assert not eng.skipped, eng.skipped
    a = check(eng, "在宅勤務手当は月額いくらですか？", "06_リモートワーク規程", "3,000円")
    assert a.page == 2 and a.source_title.endswith("p.2"), a.source_title
    a = check(eng, "在宅勤務は週何日までですか？", "06_リモートワーク規程", "週3日")
    assert a.page == 1


def run_cli_tests():
    import json
    import os
    import subprocess
    root = Path(__file__).resolve().parent.parent
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    run = lambda *a: subprocess.run([sys.executable, str(root / "ask.py"), *a], capture_output=True,  # noqa: E731
                                    text=True, encoding="utf-8", env=env)
    r = run(str(DOCS), "返品は何日以内ですか？", "--offline", "--json")
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["grounded"] and "14日" in out["excerpt"] and out["mode"] == "extractive"
    assert run(str(root / "nope")).returncode == 2
    print("CLI ok")


if __name__ == "__main__":
    run_tests()
