"""
Validation script: evaluate a checkpoint and print the three judge metrics.

Run:
    python environment/scripts/eval.py \
        --model_path checkpoints/run1 \
        --output logs/eval_run1.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from environment.datasets import load_adversarial_dataset, load_benign_dataset
from environment.judge import Judge


def evaluate(
    model_path: str,
    adversarial_data_path: str = None,
    benign_data_path: str = None,
    output_path: str = None,
):
    """
    Load model_path, run judge evaluation, and print results.

    Returns a JudgeResult dataclass. Also writes JSON to output_path if given.
    """
    print(f"[eval] Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    adv_data = load_adversarial_dataset(adversarial_data_path)
    ben_data = load_benign_dataset(benign_data_path)

    judge = Judge(adv_data, ben_data, verbose=True)
    result = judge.evaluate(model, tokenizer)

    summary = {
        "defense_success_rate": result.defense_success_rate,
        "compliance_rate": result.compliance_rate,
        "quality_score": result.quality_score,
        "reward": result.reward,
    }
    print(json.dumps(summary, indent=2))

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[eval] Results written to {output_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a hardened checkpoint.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adversarial_data", default=None)
    parser.add_argument("--benign_data", default=None)
    parser.add_argument("--output", default=None, help="Path to write JSON results")
    args = parser.parse_args()

    evaluate(args.model_path, args.adversarial_data, args.benign_data, args.output)
