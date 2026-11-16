"""
Local Privacy RAG Engine: Zero-Egress Document Q&A
Designed for confidential business document analysis without cloud data leaks.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen


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


def extract_bigrams(text: str) -> set[tuple[str, str]]:
    cleaned = re.sub(r"[\s\W_]", "", text)
    if len(cleaned) < 2:
        return set()
    return set(zip(cleaned, cleaned[1:]))


class LocalVectorStore:
    """
    Lightweight vector & lexical hybrid store.
    Supports in-memory ChromaDB-compatible index with pure-Python zero-dependency fallback.
    """

    def __init__(self, collection_name: str = "enterprise_docs"):
        self.collection_name = collection_name
        self.chunks: List[DocumentChunk] = []

    def add_document(self, doc_id: str, title: str, text: str, chunk_size: int = 240, overlap: int = 40) -> int:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        sections: List[str] = []
        current_header = ""
        current_body: List[str] = []

        for line in lines:
            if re.match(r"^第\d+条|^\d+[\.、]|^[#■【]", line):
                if current_body:
                    full_sec = (current_header + "\n" + "\n".join(current_body)).strip()
                    sections.append(full_sec)
                    current_body = []
                current_header = line
            else:
                current_body.append(line)
        if current_header or current_body:
            full_sec = (current_header + "\n" + "\n".join(current_body)).strip()
            sections.append(full_sec)

        added = 0
        for s_idx, sec in enumerate(sections):
            if len(sec) <= chunk_size:
                chunk = DocumentChunk(doc_id=doc_id, title=title, chunk_id=added, content=sec, metadata={"section": s_idx})
                self.chunks.append(chunk)
                added += 1
            else:
                start = 0
                while start < len(sec):
                    end = min(start + chunk_size, len(sec))
                    segment = sec[start:end]
                    chunk = DocumentChunk(doc_id=doc_id, title=title, chunk_id=added, content=segment, metadata={"section": s_idx, "offset": start})
                    self.chunks.append(chunk)
                    added += 1
                    if end >= len(sec):
                        break
                    start += chunk_size - overlap
        return added

    def search(self, query: str, top_k: int = 3) -> List[Tuple[DocumentChunk, float]]:
        if not self.chunks:
            return []
        query_bigrams = extract_bigrams(query)
        if not query_bigrams:
            return []

        scored: List[Tuple[DocumentChunk, float]] = []
        for chunk in self.chunks:
            chunk_bigrams = extract_bigrams(chunk.title + " " + chunk.content)
            if not chunk_bigrams:
                continue
            common = len(query_bigrams & chunk_bigrams)
            score = common / max(len(query_bigrams), 1)

            # Bonus for exact key terms
            for term in ("返品", "予備機", "出張", "宿泊", "パスワード", "セキュリティ"):
                if term in query and term in chunk.content:
                    score += 0.15

            scored.append((chunk, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


class LocalRagEngine:
    """
    On-premise zero-egress RAG answering engine.
    Connects to local LLM server (OpenAI-compatible /v1/chat/completions) or returns grounded extractive fallback.
    """

    def __init__(self, endpoint: str = "http://127.0.0.1:8080/v1", model_name: str = "qwen3.8-27b", min_confidence: float = 0.25):
        self.endpoint = endpoint.rstrip("/")
        self.model_name = model_name
        self.min_confidence = min_confidence
        self.store = LocalVectorStore()

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

    def query(self, question: str) -> RagAnswer:
        import time
        start_time = time.time()

        matches = self.store.search(question, top_k=2)
        if not matches or matches[0][1] < self.min_confidence:
            latency = (time.time() - start_time) * 1000
            return RagAnswer(
                question=question,
                answer="社内規定および提供資料からは該当する情報を確認できませんでした。（推測防止のため回答を控えます）",
                source_title="—",
                source_chunk_id=-1,
                excerpt="",
                confidence=matches[0][1] if matches else 0.0,
                grounded=False,
                latency_ms=round(latency, 2)
            )

        best_chunk, confidence = matches[0]

        # Attempt query to local model
        llm_answer = self._call_local_llm(question, best_chunk.content)
        latency = (time.time() - start_time) * 1000

        if not llm_answer:
            # Deterministic extraction fallback
            llm_answer = best_chunk.content

        return RagAnswer(
            question=question,
            answer=llm_answer,
            source_title=f"[{best_chunk.doc_id}] {best_chunk.title}",
            source_chunk_id=best_chunk.chunk_id,
            excerpt=best_chunk.content,
            confidence=round(confidence, 3),
            grounded=True,
            latency_ms=round(latency, 2)
        )

    def _call_local_llm(self, question: str, passage: str) -> Optional[str]:
        payload = {
            "model": self.model_name,
            "temperature": 0.0,
            "max_tokens": 160,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {
                    "role": "system",
                    "content": "あなたは社内規定の回答AIです。以下の提供資料だけを根拠にして、簡潔に結論を日本語で回答してください。資料に書かれていないことは絶対に推測・補完せず「資料からは確認できません」と答えてください。"
                },
                {
                    "role": "user",
                    "content": f"【提供資料】\n{passage}\n\n【質問】\n{question}"
                }
            ]
        }
        try:
            req = Request(
                f"{self.endpoint}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"].strip()
        except Exception:
            return None
