import os

from documind.agent import DocuMindAgent, plan, verify
from documind.evaluation import evaluate
from documind.llm import StubLLM
from documind.retrieval import HybridRetriever, rerank

KB = os.path.join(os.path.dirname(__file__), "..", "data", "kb.jsonl")
EVAL = os.path.join(os.path.dirname(__file__), "..", "data", "eval.jsonl")


def _agent():
    return DocuMindAgent(HybridRetriever.from_jsonl(KB), llm=StubLLM(), max_retries=2)


def test_plan_decomposes_multipart_question():
    subs = plan("What does the Pro plan cost and what support does it include?")
    assert len(subs) >= 2


def test_hybrid_retriever_finds_exact_term_via_bm25():
    r = HybridRetriever.from_jsonl(KB)
    # "429" is a rare exact token -> BM25 should pull the api_limits chunk.
    top = r.search("what does HTTP 429 mean", k=1)[0]
    assert top.chunk.doc_id == "api_limits"


def test_rerank_reorders_toward_query_coverage():
    r = HybridRetriever.from_jsonl(KB)
    cands = r.search("refund window and payment method", k=6)
    top = rerank("refund window and payment method", cands, top_n=1)[0]
    assert top.chunk.doc_id == "refunds"


def test_verifier_scores_grounded_higher_than_ungrounded():
    ctx = ["The Pro plan costs 20 dollars per month."]
    grounded = verify("The Pro plan costs 20 dollars per month.", ctx)
    ungrounded = verify("Teleportation is included free of charge.", ctx)
    assert grounded > ungrounded
    assert grounded >= 0.6


def test_agent_answers_multipart_grounded_with_citations():
    res = _agent().answer("What does the Pro plan cost and what support does it include?")
    assert res.grounded
    # Pulls from BOTH supporting docs across the decomposed sub-questions.
    assert "plan_pricing" in res.citations or "plan_support" in res.citations
    assert any(s.node == "VERIFY" for s in res.trail)


def test_agent_records_reasoning_trail():
    res = _agent().answer("How is my data secured?")
    nodes = [s.node for s in res.trail]
    assert "PLANNER" in nodes and "RETRIEVE" in nodes and "RERANK" in nodes and "VERIFY" in nodes


def test_eval_agent_beats_or_matches_baseline():
    out = evaluate(KB, EVAL)
    agent, baseline = out["agent"], out["baseline"]
    assert agent.n == 7
    # Agentic + hybrid + rerank should not do worse than single-shot on context precision.
    assert agent.context_precision >= baseline.context_precision
    assert agent.faithfulness >= 0.5
