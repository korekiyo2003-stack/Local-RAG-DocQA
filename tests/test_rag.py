"""
Test suite for Local Privacy RAG Engine.
Validates extraction accuracy, strict citation attribution, and refusal on ungrounded questions.
"""

import sys
from pathlib import Path

# Add src to path
SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from rag_engine import LocalRagEngine


def run_tests():
    docs_dir = Path(__file__).resolve().parent.parent / "samples" / "docs"
    engine = LocalRagEngine(endpoint="http://127.0.0.1:8080/v1")
    count = engine.load_directory(docs_dir)
    print(f"Indexed {count} chunks from {docs_dir}")

    # Case 1: Return window
    q1 = "返品は何日以内ですか？"
    a1 = engine.query(q1)
    print(f"\n[Q1]: {q1}\nAnswer: {a1.answer}\nSource: {a1.source_title}\nConfidence: {a1.confidence}\nLatency: {a1.latency_ms}ms")
    assert a1.grounded, "Q1 should be grounded"
    assert "14日" in a1.excerpt, "Q1 excerpt must contain '14日'"
    assert "01_返品ポリシー" in a1.source_title, "Q1 source must match 01_返品ポリシー"

    # Case 2: Equipment rental notice
    q2 = "予備機の貸出は何営業日前に申請しますか？"
    a2 = engine.query(q2)
    print(f"\n[Q2]: {q2}\nAnswer: {a2.answer}\nSource: {a2.source_title}\nConfidence: {a2.confidence}\nLatency: {a2.latency_ms}ms")
    assert a2.grounded, "Q2 should be grounded"
    assert "2営業日" in a2.excerpt, "Q2 excerpt must contain '2営業日'"
    assert "02_予備機貸出規定" in a2.source_title, "Q2 source must match 02_予備機貸出規定"

    # Case 3: Tokyo hotel cap
    q3 = "東京23区の出張宿泊費の上限はいくらですか？"
    a3 = engine.query(q3)
    print(f"\n[Q3]: {q3}\nAnswer: {a3.answer}\nSource: {a3.source_title}\nConfidence: {a3.confidence}\nLatency: {a3.latency_ms}ms")
    assert a3.grounded, "Q3 should be grounded"
    assert "12,000円" in a3.excerpt, "Q3 excerpt must contain '12,000円'"

    # Case 4: Refusal on completely ungrounded question (zero-hallucination boundary test)
    q4 = "オフィスのフリードリンクはどこに置いてありますか？"
    a4 = engine.query(q4)
    print(f"\n[Q4 (Refusal Check)]: {q4}\nAnswer: {a4.answer}\nSource: {a4.source_title}\nGrounded: {a4.grounded}")
    assert not a4.grounded, "Q4 must NOT be grounded"
    assert a4.source_title == "—", "Q4 source must be '—'"
    assert "確認できませんでした" in a4.answer or "控え" in a4.answer, "Q4 must refuse to hallucinate"

    print("\n>>> ALL 4 TEST CASES PASSED SUCCESSFULLY! <<<")


if __name__ == "__main__":
    run_tests()
