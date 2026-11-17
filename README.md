# Local-RAG-DocQA: On-Premise Document Intelligence

<div align="center">

**Zero-Cloud-Egress Enterprise Document Q&A Engine**  
*Local LLM (Qwen) · Dependency-free Japanese n-gram retrieval · Strict Source Attribution · Refusal Guard*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://python.org)
[![Privacy: 100% Offline](https://img.shields.io/badge/Privacy-100%25%20On--Premise-success.svg)](#privacy-guarantee)

[**日本語ドキュメント (README_JA.md)**](README_JA.md) • [**Architecture**](#architecture) • [**Verification Tests**](#verification-tests)

</div>

---

## 1. Overview & Business Value

In corporate environments handling confidential agreements (NDAs), HR policies, customer inquiries, and financial sheets, transmitting unredacted files to public cloud APIs violates data sovereignty rules.

**Local-RAG-DocQA** solves this through an air-gapped architecture:
* **100% Local Inference**: Integrates directly with on-premise local models (e.g. Qwen via llama-server / vLLM) on private workstations or internal servers.
* **Strict Source Attribution**: Every response cites exact file identifiers, section indices, and matched text spans.
* **Hallucination Circuit Breaker**: If document similarity falls below the verifiable threshold (e.g., questions about out-of-scope policies), the engine explicitly refuses to answer rather than inventing details.

---

## 2. Architecture

```text
[TXT / Markdown documents]
        │
        ▼ (article-aware Japanese chunker: 第N条 / numbered / ■ headings)
[Document chunks] ──► [In-memory lexical index: IDF-weighted character 2/3-grams]
                                 │
   [User query] ─────────────────┤
                                 ▼
                     [Top-3 passages + score]
                                 │
                 score < 0.03 ──►┤──► [Refusal]
                                 ▼
              [Local Qwen via llama-server, temperature 0]
              answers only from the numbered passages and cites one,
              or replies 「該当なし」 ──► [Refusal]
                                 │
                                 ▼
                [Answer + cited document + verbatim excerpt]
```

Retrieval is deliberately simple and dependency-free (no embeddings): character n-grams
work for Japanese without a tokenizer. The relevance decision is made by the local LLM, because
no lexical score separates in-scope from out-of-scope questions reliably. If no LLM is reachable,
the engine returns the best passage verbatim (`mode="extractive"`) when its score is at least 0.10,
and refuses otherwise.

---

## 3. Quickstart

### Installation
```bash
git clone https://github.com/korekiyo2003-stack/Local-RAG-DocQA.git
cd Local-RAG-DocQA
# Python 3.10+ standard library only; for answers, run any OpenAI-compatible
# local server on http://127.0.0.1:8080/v1 (e.g. llama-server with Qwen)
```

### Run Automated Verification Suite
```bash
python tests/test_rag.py
```

Sample output (Qwen on llama-server, RTX 5070 Ti):
```text
[llm] 返品は何日以内ですか？
  -> 商品受領日から14日以内です。
  source=[01_返品ポリシー] 01_返品ポリシー.txt score=0.132 365.34ms
[refused] 有給休暇は何日ありますか？
  -> 社内規定および提供資料からは該当する情報を確認できませんでした。（推測防止のため回答を控えます）
  source=— score=0.149 274.33ms
```

---

## 4. Verification & Testing Standards

`tests/test_rag.py` checks, against the sample policies in `samples/docs`:
1. **Grounding**: answers cite the right document and the excerpt holds the fact ("14日", "2営業日", "12,000円", "12文字", "役員以上").
2. **Refusal**: questions the documents do not answer are refused, including one ("有給休暇は何日ありますか？") that scores higher lexically than a real question and only the LLM can reject.
3. **Offline mode**: with no LLM reachable, the same questions get verbatim passages or a refusal.

Measured latency with the LLM is 0.3–0.7 s per question on the machine above; it depends on your model and hardware.

---

## 5. License
MIT License.
