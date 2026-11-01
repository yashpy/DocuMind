"""LoRA / QLoRA fine-tuning for domain tone (Hugging Face PEFT + TRL).

This is a *real* training script. It is guarded so the repo's test suite and CI (which have
no GPU and may not have `transformers`/`peft`/`trl`) don't try to run it — import the heavy
deps lazily inside `main()` and fail with a clear message if they're missing.

Why LoRA (and not full fine-tuning), and why fine-tune at all when we have RAG:
  * RAG supplies *fresh facts*; fine-tuning adapts *tone, format, and domain conventions*
    that are awkward to prompt for every call.
  * LoRA trains small low-rank adapters (a tiny fraction of parameters) -> roughly an order
    of magnitude less GPU memory than full fine-tuning, and the adapters are tiny and
    **hot-swappable per tenant** at serve time.
  * QLoRA (4-bit base + LoRA) pushes this further so a 7-8B model fine-tunes on a single
    consumer GPU.

Run:
    python training/finetune_lora.py \
        --base meta-llama/Meta-Llama-3-8B-Instruct \
        --data data/sft.jsonl --output out/adapter --qlora
"""

from __future__ import annotations

import argparse


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LoRA/QLoRA SFT for domain tone.")
    p.add_argument("--base", required=True, help="base model id (HF hub)")
    p.add_argument("--data", required=True, help="JSONL with {prompt, completion} rows")
    p.add_argument("--output", default="out/adapter")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16, help="LoRA rank r")
    p.add_argument("--alpha", type=int, default=32, help="LoRA alpha")
    p.add_argument("--qlora", action="store_true", help="4-bit base (QLoRA)")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:  # pragma: no cover - requires the ML stack + GPU
        raise SystemExit(
            "This script needs the training stack: "
            "pip install 'transformers>=4.40' peft trl datasets bitsandbytes accelerate torch\n"
            f"(missing: {exc.name})"
        )

    quant = None
    if args.qlora:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        quantization_config=quant,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    dataset = load_dataset("json", data_files=args.data, split="train")

    def format_row(row):
        return {"text": f"{row['prompt']}\n{row['completion']}{tokenizer.eos_token}"}

    dataset = dataset.map(format_row)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora,
        args=SFTConfig(
            output_dir=args.output,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            logging_steps=10,
            bf16=True,
            dataset_text_field="text",
        ),
    )
    trainer.train()
    trainer.save_model(args.output)  # saves the small LoRA adapter only
    print(f"Saved LoRA adapter to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
