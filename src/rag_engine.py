"""
Local Privacy RAG Engine: Zero-Egress Document Q&A
Designed for confidential business document analysis without cloud data leaks.

Retrieval is lexical (IDF-weighted character 2/3-grams, which work for Japanese
without a tokenizer). The local LLM reads the top passages, answers only from
them and names the passage it used, or says the answer is not there. Without a
reachable LLM the engine falls back to returning the best passage verbatim.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen

REFUSAL = "社内規定および提供資料からは該当する情報を確認できませんでした。（推測防止のため回答を控えます）"
NOT_FOUND_TOKEN = "該当なし"


@dataclass
class DocumentChunk:
    doc_id: str
    title: str
    chunk_id: int
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RagAnswer:
    question: str
    answer: str
    source_title: str
    source_chunk_id: int
    excerpt: str
    confidence: float
    grounded: bool
    latency_ms: float = 0.0
    mode: str = "llm"  # "llm", "extractive" (no LLM reachable) or "refused"


def extract_ngrams(text: str, sizes: Tuple[int, ...] = (2, 3)) -> set[str]:
    """Character n-grams with spaces and punctuation removed."""
    cleaned = re.sub(r"[\s\W_]", "", text)
    grams: set[str] = set()
    for n in sizes:
        grams.update(cleaned[i:i + n] for i in range(len(cleaned) - n + 1))
    return grams


class LexicalStore:
    """In-memory store ranked by IDF-weighted character n-gram overlap."""

    def __init__(self, collection_name: str = "enterprise_docs"):
        self.collection_name = collection_name
        self.chunks: List[DocumentChunk] = []
        self._grams: List[set[str]] = []
        self._df: Dict[str, int] = {}

    def add_document(self, doc_id: str, title: str, text: str, chunk_size: int = 240, overlap: int = 40) -> int:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        sections: List[str] = []
        current_header = ""
        current_body: List[str] = []

        for line in lines:
            if re.match(r"^第\d+条|^\d+[\.、]|^[#■【]", line):
                if current_header or current_body:
                    sections.append((current_header + "\n" + "\n".join(current_body)).strip())
                    current_body = []
                current_header = line
            else:
                current_body.append(line)
        if current_header or current_body:
            sections.append((current_header + "\n" + "\n".join(current_body)).strip())

        added = 0
        for s_idx, sec in enumerate(sections):
            # Header line of the article, repeated on every piece of a long section
            header = sec.split("\n", 1)[0]
            start = 0
            while True:
                end = min(start + chunk_size, len(sec))
                segment = sec[start:end]
                if start > 0:
                    segment = header + "\n…" + segment
                self._add_chunk(DocumentChunk(doc_id=doc_id, title=title, chunk_id=added, content=segment,
                                              metadata={"section": s_idx, "offset": start}))
                added += 1
                if end >= len(sec):
                    break
                start += chunk_size - overlap
        return added

    def _add_chunk(self, chunk: DocumentChunk) -> None:
        grams = extract_ngrams(chunk.content)
        self.chunks.append(chunk)
        self._grams.append(grams)
        for g in grams:
            self._df[g] = self._df.get(g, 0) + 1

    def _idf(self, gram: str) -> float:
        n = len(self.chunks)
        return math.log((n + 1) / (self._df.get(gram, 0) + 0.5))

    def search(self, query: str, top_k: int = 3) -> List[Tuple[DocumentChunk, float]]:
        """Score = IDF weight of the query n-grams found in the chunk / IDF weight of all query n-grams (0..1)."""
        if not self.chunks:
            return []
        query_grams = extract_ngrams(query)
        total = sum(self._idf(g) for g in query_grams)
        if total <= 0:
            return []
        scored = [
            (chunk, sum(self._idf(g) for g in query_grams & grams) / total)
            for chunk, grams in zip(self.chunks, self._grams)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s in scored[:top_k] if s[1] > 0]


# Kept for code written against the first version of this module.
LocalVectorStore = LexicalStore


class LocalRagEngine:
    """
    On-premise zero-egress RAG answering engine.
    Talks to a local OpenAI-compatible server (llama-server, vLLM, Ollama) at `endpoint`.
    """

    def __init__(self, endpoint: str = "http://127.0.0.1:8080/v1", model_name: str = "qwen3.8-27b",
                 min_score: float = 0.03, offline_min_score: float = 0.10, top_k: int = 3, timeout: float = 30.0):
        self.endpoint = endpoint.rstrip("/")
        self.model_name = model_name
        self.min_score = min_score                  # below this nothing is even sent to the LLM
        self.offline_min_score = offline_min_score  # stricter bar when no LLM can judge relevance
        self.top_k = top_k
        self.timeout = timeout
        self.store = LexicalStore()

    def load_directory(self, folder: str | Path) -> int:
        path = Path(folder)
        if not path.exists():
            return 0
        total_chunks = 0
        for f in sorted(path.glob("*.*")):
            if f.suffix.lower() in (".txt", ".md"):
                text = f.read_text(encoding="utf-8-sig")
                total_chunks += self.store.add_document(doc_id=f.stem, title=f.name, text=text)
        return total_chunks

    def _refuse(self, question: str, score: float, t0: float) -> RagAnswer:
        return RagAnswer(question=question, answer=REFUSAL, source_title="—", source_chunk_id=-1, excerpt="",
                         confidence=round(score, 3), grounded=False,
                         latency_ms=round((time.time() - t0) * 1000, 2), mode="refused")

    def _answer(self, question: str, text: str, chunk: DocumentChunk, score: float, t0: float, mode: str) -> RagAnswer:
        return RagAnswer(question=question, answer=text, source_title=f"[{chunk.doc_id}] {chunk.title}",
                         source_chunk_id=chunk.chunk_id, excerpt=chunk.content, confidence=round(score, 3),
                         grounded=True, latency_ms=round((time.time() - t0) * 1000, 2), mode=mode)

    def query(self, question: str) -> RagAnswer:
        t0 = time.time()
        matches = self.store.search(question, top_k=self.top_k)
        if not matches or matches[0][1] < self.min_score:
            return self._refuse(question, matches[0][1] if matches else 0.0, t0)

        reply = self._call_local_llm(question, [c.content for c, _ in matches])
        if reply is None:
            # No LLM: return the best passage verbatim, but only on a clearer lexical match.
            best, score = matches[0]
            if score < self.offline_min_score:
                return self._refuse(question, score, t0)
            return self._answer(question, best.content, best, score, t0, mode="extractive")

        answer, source_no = self._parse_reply(reply, len(matches))
        if answer is None:
            return self._refuse(question, matches[0][1], t0)
        chunk, score = matches[source_no - 1]
        return self._answer(question, answer, chunk, score, t0, mode="llm")

    @staticmethod
    def _parse_reply(reply: str, n_passages: int) -> Tuple[Optional[str], int]:
        """Return (answer, 1-based passage number) or (None, 0) when the model found nothing."""
        text = reply.strip()
        if not text or text.startswith(NOT_FOUND_TOKEN) or "資料からは確認できません" in text:
            return None, 0
        m = re.search(r"[\[［](\d+)[\]］]", text)
        source_no = int(m.group(1)) if m else 1
        if not 1 <= source_no <= n_passages:
            source_no = 1
        answer = re.sub(r"\s*[\[［]\d+[\]］]\s*", "", text).strip()
        return (answer or None), source_no

    def _call_local_llm(self, question: str, passages: List[str]) -> Optional[str]:
        numbered = "\n\n".join(f"[{i}]\n{p}" for i, p in enumerate(passages, start=1))
        payload = {
            "model": self.model_name,
            "temperature": 0.0,
            "max_tokens": 200,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "あなたは社内規定の回答AIです。番号付きの提供資料だけを根拠に、日本語で簡潔に結論を答えてください。"
                        "回答の末尾に根拠にした資料の番号を [1] の形で1つだけ付けてください。"
                        f"資料に答えが書かれていない場合は推測・補完せず、「{NOT_FOUND_TOKEN}」とだけ出力してください。"
                    ),
                },
                {"role": "user", "content": f"【提供資料】\n{numbered}\n\n【質問】\n{question}"},
            ],
        }
        try:
            req = Request(
                f"{self.endpoint}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return (data["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            return None
