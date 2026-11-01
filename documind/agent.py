"""The agentic RAG graph: PLANNER -> RETRIEVE -> RERANK -> DRAFT -> VERIFY (self-correct).

This is the core idea of DocuMind. Rather than a single retrieve-then-answer shot, the agent
runs a small state machine with a **verifier** that grades its own draft for grounding and,
if the draft isn't supported by the evidence, loops back to retrieve MORE context (widening
k) and redraft — up to a bounded retry budget. Every run records a ReasoningTrail for
observability.

The control flow mirrors a LangGraph state graph; it's implemented in plain Python here so
the repo has zero heavy deps and runs anywhere, but the node/edge structure maps 1:1 onto
LangGraph nodes.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from typing import List, Optional

from .llm import LLM, get_llm
from .retrieval import HybridRetriever, Scored, rerank, tokens

SYSTEM = (
    "You are a precise domain expert. Answer the QUESTION using ONLY the CONTEXT. "
    "Cite nothing you cannot support. If the context is insufficient, say so."
)


@dataclass
class Step:
    node: str
    detail: str


@dataclass
class AgentResult:
    answer: str
    citations: List[str] = field(default_factory=list)
    grounded: bool = False
    attempts: int = 0
    trail: List[Step] = field(default_factory=list)


def plan(question: str) -> List[str]:
    """Decompose a multi-part question into sub-questions on 'and'/'?' boundaries."""
    parts = re.split(r"\band\b|\?|;", question, flags=re.IGNORECASE)
    subs = [p.strip() for p in parts if len(p.strip()) > 3]
    return subs or [question.strip()]


def verify(answer: str, contexts: List[str], min_support: float = 0.6) -> float:
    """Grounding score: fraction of answer content-tokens supported by the context.

    This is the critic. A low score means the draft asserts things the evidence doesn't
    support -> the agent should retrieve more and retry rather than return it.
    """
    a = set(tokens(answer))
    if not a:
        return 0.0
    ctx = set()
    for c in contexts:
        ctx |= set(tokens(c))
    return len(a & ctx) / len(a)


class DocuMindAgent:
    def __init__(self, retriever: HybridRetriever, llm: Optional[LLM] = None, max_retries: int = 2):
        self.retriever = retriever
        self.llm = llm or get_llm()
        self.max_retries = max_retries

    def answer(self, question: str) -> AgentResult:
        trail: List[Step] = []
        subs = plan(question)
        trail.append(Step("PLANNER", f"{len(subs)} sub-question(s): {subs}"))

        k = 4
        best = AgentResult(answer="", grounded=False)
        for attempt in range(1, self.max_retries + 2):  # initial try + retries
            # Retrieve across all sub-questions, dedupe by doc_id, then rerank.
            pool: List[Scored] = []
            seen = set()
            for sub in subs:
                for s in self.retriever.search(sub, k=k):
                    if s.chunk.doc_id not in seen:
                        seen.add(s.chunk.doc_id)
                        pool.append(s)
            trail.append(Step("RETRIEVE", f"attempt {attempt}: k={k}, {len(pool)} unique chunks"))

            top = rerank(question, pool, top_n=min(3, len(pool)))
            contexts = [s.chunk.text for s in top]
            citations = [s.chunk.doc_id for s in top]
            trail.append(Step("RERANK", f"kept {citations}"))

            ctx_block = "CONTEXT:\n" + "\n".join(f"- {c}" for c in contexts)
            if len(subs) > 1:
                # Answer each sub-question against the shared context, then compose —
                # this is where decomposition pays off on multi-part questions.
                parts = [
                    self.llm.generate(SYSTEM, f"{ctx_block}\n\nQUESTION:\n{sub}")
                    for sub in subs
                ]
                seen_parts: List[str] = []
                for part in parts:
                    if part and "don't have enough information" not in part and part not in seen_parts:
                        seen_parts.append(part)
                draft = " ".join(seen_parts) or self.llm.generate(SYSTEM, f"{ctx_block}\n\nQUESTION:\n{question}")
            else:
                draft = self.llm.generate(SYSTEM, f"{ctx_block}\n\nQUESTION:\n{question}")
            score = verify(draft, contexts)
            trail.append(Step("VERIFY", f"grounding={score:.2f}"))

            result = AgentResult(
                answer=draft, citations=citations, grounded=score >= 0.6,
                attempts=attempt, trail=list(trail),
            )
            if result.grounded:
                return result

            # Self-correct: widen retrieval and try again.
            best = result
            k += 3
            trail.append(Step("SELF_CORRECT", f"grounding below threshold; widening k to {k}"))

        return best


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="DocuMind agentic RAG.")
    p.add_argument("--kb", required=True)
    p.add_argument("--ask", required=True)
    p.add_argument("--trace", action="store_true", help="print the reasoning trail")
    args = p.parse_args(argv)

    agent = DocuMindAgent(HybridRetriever.from_jsonl(args.kb))
    res = agent.answer(args.ask)
    if args.trace or True:
        print("--- reasoning trail ---")
        for step in res.trail:
            print(f"  [{step.node}] {step.detail}")
    print("\nAnswer    :", res.answer)
    print("Citations :", ", ".join(res.citations))
    print(f"Grounded  : {res.grounded}  (attempts={res.attempts})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
