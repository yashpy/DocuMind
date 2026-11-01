# DocuMind

**A self-correcting, agentic domain-expert RAG platform — it decomposes a hard question,
retrieves with a hybrid lexical+semantic search, drafts an answer, then *verifies its own
answer* against the evidence and retries until it's grounded. Plus a LoRA fine-tuning track
for domain tone and a RAGAS-style evaluation harness.**

Single-shot RAG (`retrieve top-k → stuff into prompt → answer`) falls over on complex,
multi-part questions and hallucinates confidently. DocuMind is built like an **agent with a
critic**: it plans, retrieves, answers, and then checks whether its own answer is actually
supported by the retrieved evidence — looping back to retrieve more if not.

```
                         ┌─────────────────────────────────────────────┐
  question ──▶ PLANNER ──▶ decompose into sub-questions                  │
                         │        │                                      │
                         │        ▼                                      │
                         │   HYBRID RETRIEVE  (BM25 ⊕ dense embeddings)   │
                         │        │                                      │
                         │        ▼                                      │
                         │     RERANK  (cross-encoder-style scoring)     │
                         │        │                                      │
                         │        ▼                                      │
                         │     DRAFT  (grounded answer + citations)      │
                         │        │                                      │
                         │        ▼                                      │
                         │   VERIFY  ──grounded?──┐                      │
                         │        │ no            │ yes                  │
                         └────────┘ (retry w/ ────┘──▶ final answer      │
                                      more context, up to N)             │
                         └─────────────────────────────────────────────┘
```

## Why it matters & what makes it strong

- **Agentic, not single-shot.** A `PLANNER → RETRIEVE → RERANK → DRAFT → VERIFY` graph with a
  **self-correction loop** — the verifier grades grounding and sends the agent back to
  retrieve more evidence instead of returning a confident-but-wrong answer.
- **Hybrid retrieval.** Combines **BM25** (lexical, great for exact terms/IDs) with **dense
  embeddings** (semantic), then reranks — consistently beats either alone.
- **Grounded with citations.** Every claim is tied to a source chunk; ungrounded drafts are
  rejected by the verifier.
- **Two paths to domain quality — the right tool for each job.** **RAG** supplies fresh
  facts; **LoRA/QLoRA fine-tuning** (`training/`) adapts *tone/format* cheaply (small
  adapters, ~an order of magnitude less GPU than full fine-tuning, hot-swappable per tenant).
- **Measured, not vibes.** A **RAGAS-style eval harness** (faithfulness, answer-relevance,
  context-precision/recall) gates changes in CI.
- **Runs with no API key / no GPU.** A deterministic offline LLM stub drives the agent and
  eval so the whole graph is testable in CI; real providers and the PEFT trainer plug in
  behind the same interfaces.

## What's in here

| Path | What it is |
| --- | --- |
| `documind/agent.py` | The agentic graph: planner, hybrid retrieve, rerank, draft, **verifier + self-correction loop**. |
| `documind/retrieval.py` | BM25 + dense hybrid retriever and a cross-encoder-style reranker. |
| `documind/llm.py` | Provider interface (OpenAI / Groq / offline deterministic stub). |
| `documind/evaluation.py` | RAGAS-style metrics: faithfulness, answer-relevance, context precision/recall. |
| `training/finetune_lora.py` | A real LoRA/QLoRA fine-tuning script (Hugging Face PEFT + TRL), guarded so it only runs where `transformers`/`peft` + a GPU are available. |
| `training/README.md` | The fine-tuning rationale and how to run it. |
| `data/` | A small domain KB and a labeled eval set. |
| `tests/` | Unit tests for retrieval, the verifier/self-correction loop, and the metrics. |

## Run it (no API key, no GPU)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Ask the agent a multi-part question and watch the reasoning trail
python -m documind.agent --kb data/kb.jsonl --ask "What does the Pro plan cost and what support does it include?"

# Run the RAGAS-style evaluation
python -m documind.evaluation --kb data/kb.jsonl --eval data/eval.jsonl
```

## My contribution

I designed and built the platform end to end:

- **The agentic graph with a self-correcting verifier** — the core idea: a critic step that
  grades its own draft for grounding and loops back to retrieve more evidence, with a bounded
  retry budget. This is what turns RAG from a demo into something that refuses to hallucinate.
- **Hybrid retrieval + reranking** — BM25 fused with dense embeddings and a reranking pass,
  with the fusion weighting exposed and tested.
- **The RAGAS-style eval harness** — faithfulness, answer-relevance, and context
  precision/recall, used to validate that the agent + hybrid retrieval beat a single-shot
  baseline (`documind.evaluation` reports both).
- **The LoRA fine-tuning track** — a real PEFT/TRL training script and the rationale for
  choosing LoRA (cheap, swappable adapters) over full fine-tuning, and RAG for fresh facts.

## Results / evaluation

`documind.evaluation` reports the RAGAS-style metrics for the agent and prints the
single-shot baseline alongside, so the lift from agentic + hybrid retrieval is reproducible.
Resume figures come from running this harness on a fuller labeled set; the repo ships the
harness and a seed set rather than hard-coded numbers.

## License

MIT
