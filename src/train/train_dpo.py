from __future__ import annotations

import argparse
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer


def load_model(model_path: str, load_in_4bit: bool):
    quant_cfg = None
    if load_in_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    return AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quant_cfg,
        dtype=torch.bfloat16,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA DPO training entrypoint.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-pairs", required=True)
    parser.add_argument("--eval-pairs", required=False, default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--grad-acc", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--max-prompt-len", type=int, default=768)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--load-in-4bit", action="store_true", default=False)
    args = parser.parse_args()

    model = load_model(args.model_path, load_in_4bit=args.load_in_4bit)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_dataset("json", data_files=args.train_pairs, split="train")
    eval_dataset: Optional[object] = None
    if args.eval_pairs:
        eval_dataset = load_dataset("json", data_files=args.eval_pairs, split="train")

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    train_args = DPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_acc,
        max_length=args.max_len,
        max_prompt_length=args.max_prompt_len,
        beta=args.beta,
        bf16=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset is not None else "no",
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
