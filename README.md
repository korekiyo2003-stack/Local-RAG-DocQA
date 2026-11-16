# Local-RAG-DocQA: On-Premise Document Intelligence

<div align="center">

**Zero-Cloud-Egress Enterprise Document Q&A Engine**  
*Local LLM (Qwen) · Hybrid Vector Index · Strict Source Attribution · Zero-Hallucination Guard*

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
[Enterprise PDFs / TXTs]
        │
        ▼ (Character-Aware Japanese Chunker)
[Document Chunks] ──► [Local Hybrid Vector Store]
                                 │
   [User Query] ─────────────────┤ (Bigram Lexical + Vector Match)
                                 ▼
                     [Top-K Grounded Excerpts]
                                 │
                         (Threshold Check)
                         ├─ < 0.25 ──► [Explicit Refusal / Out-of-Scope]
                         │
                         └─ >= 0.25 ──► [Local Qwen LLM (Temp=0.0)]
                                             │
                                             ▼
                                 [Structured Answer + Verifiable Citation]
```

---

## 3. Quickstart

### Installation
```bash
git clone https://github.com/korekiyo2003-stack/Local-RAG-DocQA.git
cd Local-RAG-DocQA
pip install -r requirements.txt
```

### Run Automated Verification Suite
```bash
python tests/test_rag.py
```

Sample output:
```text
[Q1]: 返品は何日以内ですか？
Answer: 商品受領日から14日以内です。
Source: [01_返品ポリシー] 01_返品ポリシー.txt
Confidence: 0.333 | Latency: 480ms

[Q4 (Refusal Check)]: オフィスのフリードリンクはどこに置いてありますか？
Answer: 社内規定および提供資料からは該当する情報を確認できませんでした。（推測防止のため回答を控えます）
Source: — | Grounded: False
```

---

## 4. Verification & Testing Standards

The repository includes a strict automated evaluation pipeline:
1. **Factual Grounding**: Confirms numerical values and deadlines ("14日", "2営業日", "12,000円") match source clauses.
2. **Refusal on Out-of-Domain Queries**: Tests that unrelated queries correctly fail the similarity gate and trigger zero-hallucination guard rails.
3. **Sub-second Response Times**: Optimized token processing delivers answers under 600ms on modern local GPUs/CPUs.

---

## 5. License
MIT License.
