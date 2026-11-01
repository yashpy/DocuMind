"""Hybrid retrieval: BM25 (lexical) fused with dense embeddings (semantic), then rerank.

Lexical search nails exact terms, IDs, and rare tokens; dense search captures paraphrase
and intent. Fusing them and reranking beats either alone — especially on domain questions
that mix precise terms with natural phrasing.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

_TOKEN = re.compile(r"[a-z0-9]+")
_DIM = 1024
_STOP = {
    "the", "a", "an", "is", "are", "to", "of", "and", "or", "in", "on", "for",
    "do", "i", "you", "how", "what", "when", "within", "my", "it", "that", "this",
    "can", "get", "with", "be", "as", "at", "by", "from", "have", "has", "does",
}


def tokens(text: str) -> List[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def _bucket(tok: str, dim: int = _DIM) -> int:
    return int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big") % dim


def embed(text: str, dim: int = _DIM) -> List[float]:
    vec = [0.0] * dim
    for t in tokens(text):
        vec[_bucket(t, dim)] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass
class Chunk:
    doc_id: str
    text: str
    emb: List[float]
    tf: Counter


@dataclass
class Scored:
    chunk: Chunk
    score: float


class HybridRetriever:
    """BM25 ⊕ dense fusion. `alpha` weights dense vs. lexical (0..1)."""

    def __init__(self, alpha: float = 0.5, k1: float = 1.5, b: float = 0.75):
        self.alpha, self.k1, self.b = alpha, k1, b
        self.chunks: List[Chunk] = []
        self._df: Counter = Counter()
        self._avg_len = 0.0

    @classmethod
    def from_jsonl(cls, path: str, **kw) -> "HybridRetriever":
        r = cls(**kw)
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    r.add(rec["doc_id"], rec["text"])
        r.finalize()
        return r

    def add(self, doc_id: str, text: str) -> None:
        tf = Counter(tokens(text))
        self.chunks.append(Chunk(doc_id, text, embed(text), tf))

    def finalize(self) -> None:
        self._df = Counter()
        total = 0
        for c in self.chunks:
            total += sum(c.tf.values())
            for term in c.tf:
                self._df[term] += 1
        self._avg_len = total / max(len(self.chunks), 1)

    def _bm25(self, q_terms: List[str], c: Chunk) -> float:
        n = len(self.chunks)
        length = sum(c.tf.values()) or 1
        score = 0.0
        for term in q_terms:
            if term not in c.tf:
                continue
            df = self._df.get(term, 0) or 1
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            tf = c.tf[term]
            denom = tf + self.k1 * (1 - self.b + self.b * length / self._avg_len)
            score += idf * (tf * (self.k1 + 1)) / denom
        return score

    @staticmethod
    def _normalize(scores: List[float]) -> List[float]:
        if not scores:
            return scores
        lo, hi = min(scores), max(scores)
        if hi - lo < 1e-12:
            return [0.0 for _ in scores]
        return [(s - lo) / (hi - lo) for s in scores]

    def search(self, query: str, k: int = 4) -> List[Scored]:
        q_terms = tokens(query)
        q_emb = embed(query)
        bm25 = [self._bm25(q_terms, c) for c in self.chunks]
        dense = [cosine(q_emb, c.emb) for c in self.chunks]
        bm25_n, dense_n = self._normalize(bm25), self._normalize(dense)
        fused = [
            Scored(c, self.alpha * d + (1 - self.alpha) * l)
            for c, d, l in zip(self.chunks, dense_n, bm25_n)
        ]
        fused.sort(key=lambda s: s.score, reverse=True)
        return fused[:k]


def rerank(query: str, candidates: List[Scored], top_n: int = 3) -> List[Scored]:
    """Cross-encoder-style rerank: re-score query↔chunk jointly by term coverage + density.

    A lightweight stand-in for a cross-encoder that still meaningfully reorders candidates
    toward the ones that actually cover the query terms; swap for a real cross-encoder in
    production behind this same signature.
    """
    q = set(tokens(query))
    rescored: List[Scored] = []
    for cand in candidates:
        c_terms = tokens(cand.chunk.text)
        if not c_terms:
            rescored.append(Scored(cand.chunk, 0.0))
            continue
        covered = len(q & set(c_terms)) / max(len(q), 1)
        density = len(q & set(c_terms)) / len(c_terms)
        rescored.append(Scored(cand.chunk, 0.7 * covered + 0.3 * density))
    rescored.sort(key=lambda s: s.score, reverse=True)
    return rescored[:top_n]
