# DocuMind fine-tuning (LoRA / QLoRA)

RAG and fine-tuning solve **different** problems, and DocuMind uses both deliberately:

| Need | Solution | Why |
| --- | --- | --- |
| Up-to-date, verifiable facts | **RAG** (the agent) | Facts change; retrieval keeps them fresh and citable without retraining. |
| Domain tone, format, house style | **LoRA/QLoRA fine-tune** | Hard to prompt for consistently; cheap to bake into a small adapter. |

## Why LoRA over full fine-tuning

- Trains small low-rank adapters (a tiny fraction of the model's parameters) → roughly an
  order of magnitude less GPU memory than full fine-tuning.
- Adapters are a few MB and **hot-swappable per tenant** at serve time — one base model,
  many domain adapters.
- **QLoRA** (4-bit base + LoRA) lets a 7–8B model fine-tune on a single consumer GPU.

## Data format (`data/sft.jsonl`)

```json
{"prompt": "Rewrite in our support voice: your refund is coming", "completion": "Your refund is on its way and should arrive within 5–7 business days."}
```

## Run

```bash
pip install "transformers>=4.40" peft trl datasets bitsandbytes accelerate torch

python training/finetune_lora.py \
  --base meta-llama/Meta-Llama-3-8B-Instruct \
  --data data/sft.jsonl \
  --output out/adapter \
  --qlora --rank 16 --alpha 32
```

The script is guarded: on a machine without the training stack it exits with a clear
install message instead of failing obscurely, so the rest of the repo (agent + eval) stays
runnable everywhere.

## Evaluating the adapter

Point an OpenAI-compatible server (e.g. vLLM) at `base + adapter` and run
`python -m documind.evaluation` with `DOCUMIND_PROVIDER` set to that endpoint to compare the
fine-tuned model's win-rate against the base model on the labeled set.
