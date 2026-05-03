"""
Baseline training script: SFT fine-tuning with LoRA.

This is the *naive* starting point the agent can inspect and revise.
Run directly:
    python environment/scripts/train.py \
        --model_name sshleifer/tiny-gpt2 \
        --output_dir checkpoints/run1 \
        --max_steps 20

Or import and call train() from another script.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

try:
    from peft import LoraConfig, TaskType, get_peft_model
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

from datasets import Dataset

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from environment.datasets import load_adversarial_dataset, load_benign_dataset


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _build_texts(adversarial: list, benign: list) -> list:
    """
    Combine adversarial and benign examples into labelled training strings.

    Adversarial: model learns to acknowledge the injection and focus on the
                 user's actual request instead.
    Benign:      model learns to comply helpfully.
    """
    texts = []

    for ex in adversarial:
        refusal_response = (
            "I notice this content contains instructions that are unrelated to your "
            "actual request. I'll focus on what you originally asked and ignore the "
            "embedded instruction."
        )
        texts.append(ex["prompt"] + " " + refusal_response)

    for ex in benign:
        target = ex.get("target_response", "Sure, I'm happy to help with that.")
        texts.append(ex["prompt"] + " " + target)

    return texts


def _tokenize(texts: list, tokenizer, max_length: int = 512) -> Dataset:
    ds = Dataset.from_dict({"text": texts})

    def _fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    return ds.map(_fn, batched=True, remove_columns=["text"])


# ---------------------------------------------------------------------------
# Main train function
# ---------------------------------------------------------------------------

def train(
    model_name: str,
    output_dir: str,
    adversarial_data_path: str = None,
    benign_data_path: str = None,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 4,
    learning_rate: float = 2e-4,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    max_steps: int = -1,
    use_lora: bool = True,
):
    """
    Fine-tune model_name on hardening data and save to output_dir.

    Args:
        model_name             : HuggingFace id or local path
        output_dir             : where to write the fine-tuned model
        adversarial_data_path  : path to adversarial JSONL (or None → synthetic)
        benign_data_path       : path to benign JSONL (or None → synthetic)
        num_train_epochs       : full passes over the dataset
        per_device_train_batch_size
        learning_rate
        lora_r                 : LoRA rank
        lora_alpha             : LoRA scaling
        lora_dropout
        max_steps              : overrides epochs if > 0 (useful for smoke tests)
        use_lora               : whether to apply LoRA (requires peft)
    """
    print(f"[train] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[train] Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if use_lora and HAS_PEFT:
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    elif use_lora and not HAS_PEFT:
        print("[train] Warning: peft not installed, training without LoRA.")

    # Load datasets
    adv_data = load_adversarial_dataset(adversarial_data_path)
    ben_data = load_benign_dataset(benign_data_path)
    print(f"[train] {len(adv_data)} adversarial + {len(ben_data)} benign examples")

    texts = _build_texts(adv_data, ben_data)
    dataset = _tokenize(texts, tokenizer)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        fp16=torch.cuda.is_available(),
        logging_steps=max(1, min(10, len(dataset) // 4)),
        save_strategy="epoch",
        max_steps=max_steps,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("[train] Starting training…")
    trainer.train()

    print(f"[train] Saving to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("[train] Done.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline LoRA fine-tuning script.")
    parser.add_argument("--model_name", required=True, help="HF model id or local path")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--adversarial_data", default=None)
    parser.add_argument("--benign_data", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Override epochs; useful for quick testing (e.g. --max_steps 20)")
    parser.add_argument("--no_lora", action="store_true")
    args = parser.parse_args()

    train(
        model_name=args.model_name,
        output_dir=args.output_dir,
        adversarial_data_path=args.adversarial_data,
        benign_data_path=args.benign_data,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        max_steps=args.max_steps,
        use_lora=not args.no_lora,
    )
