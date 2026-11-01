"""RAGAS-style evaluation harness.

Reports four metrics in [0,1] and prints the single-shot baseline alongside the agent so
the lift from agentic + hybrid retrieval is visible:

  * faithfulness        — answer tokens supported by retrieved context
  * answer_relevance    — overlap of answer with the expected answer
  * context_precision   — fraction of retrieved chunks that are actually relevant
  * context_recall      — fraction of the expected supporting chunks that were retrieved
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from typing import List

from .agent import DocuMindAgent, verify
from .retrieval import HybridRetriever


def _tok(s: str) -> set:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


@dataclass
class Metrics:
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float
    n: int

    def render(self, label: str) -> str:
        return (
            f"[{label}] n={self.n}  "
            f"faithfulness={self.faithfulness:.2f}  "
            f"answer_relevance={self.answer_relevance:.2f}  "
            f"context_precision={self.context_precision:.2f}  "
            f"context_recall={self.context_recall:.2f}"
        )


def _answer_relevance(answer: str, expected: str) -> float:
    e = _tok(expected)
    return len(_tok(answer) & e) / len(e) if e else 0.0


def evaluate(kb_path: str, eval_path: str) -> dict:
    retriever = HybridRetriever.from_jsonl(kb_path)
    agent = DocuMindAgent(retriever)

    cases = []
    with open(eval_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    def run(agentic: bool) -> Metrics:
        faith = rel = prec = rec = 0.0
        for case in cases:
            q = case["question"]
            expected_docs = set(case.get("supporting_docs", []))

            if agentic:
                res = agent.answer(q)
                retrieved = res.citations
                answer = res.answer
                contexts = [c.chunk.text for c in retriever.search(q, k=len(retrieved) or 3)]
            else:
                # Single-shot baseline: top-k dense/lexical, no rerank, no verify/retry.
                hits = retriever.search(q, k=3)
                retrieved = [h.chunk.doc_id for h in hits]
                contexts = [h.chunk.text for h in hits]
                answer = agent.llm.generate(
                    "Answer from context.",
                    "CONTEXT:\n" + "\n".join(f"- {c}" for c in contexts) + f"\n\nQUESTION:\n{q}",
                )

            retrieved_set = set(retrieved)
            faith += verify(answer, contexts)
            rel += _answer_relevance(answer, case.get("expected_answer", ""))
            prec += (len(retrieved_set & expected_docs) / len(retrieved_set)) if retrieved_set else 0.0
            rec += (len(retrieved_set & expected_docs) / len(expected_docs)) if expected_docs else 0.0

        d = max(len(cases), 1)
        return Metrics(faith / d, rel / d, prec / d, rec / d, len(cases))

    return {"agent": run(True), "baseline": run(False)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="DocuMind RAGAS-style evaluation.")
    p.add_argument("--kb", required=True)
    p.add_argument("--eval", required=True)
    args = p.parse_args(argv)
    out = evaluate(args.kb, args.eval)
    print(out["baseline"].render("single-shot baseline"))
    print(out["agent"].render("agentic + hybrid    "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
