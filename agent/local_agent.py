"""
agent/local_agent.py — local HuggingFace causal LM agent.

This is the agent updated by training/rl_train.py via REINFORCE.
compute_log_probs() is called by the RL training loop to compute
the policy gradient loss.
"""
from __future__ import annotations

from agent.base import BaseAgent, SYSTEM_PROMPT


class LocalModelAgent(BaseAgent):
    """
    Agent backed by a local HuggingFace causal LM.

    Args:
        model_name_or_path : HF model id or local path
        max_new_tokens     : token budget for the generated action
        device             : "cpu", "cuda", or "auto"
        load_in_8bit       : quantise to 8-bit to reduce VRAM usage
    """

    def __init__(
        self,
        model_name_or_path: str,
        max_new_tokens: int = 256,
        device: str = "auto",
        load_in_8bit: bool = False,
        max_history_turns: int = 20,
    ):
        super().__init__(max_history_turns=max_history_turns)
        self.max_new_tokens = max_new_tokens
        self.model_name_or_path = model_name_or_path

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[LocalModelAgent] Loading {model_name_or_path}…")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = {}
        if load_in_8bit:
            kwargs["load_in_8bit"] = True
        elif torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device if torch.cuda.is_available() else None,
            **kwargs,
        )
        self.model.eval()

    def _call_model(self, messages: list[dict]) -> str:
        import torch

        prompt = self._format_prompt(messages)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    def compute_log_probs(self, messages: list[dict], action: str) -> "torch.Tensor":
        """
        Compute the sum of log-probabilities for action tokens given messages.
        Used by the RL training loop to compute the policy gradient loss.
        Returns a scalar tensor with grad_fn attached.
        """
        import torch

        prompt = self._format_prompt(messages)
        full_text = prompt + action

        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        full_ids = self.tokenizer(full_text, return_tensors="pt")["input_ids"]

        prompt_ids = prompt_ids.to(self.model.device)
        full_ids = full_ids.to(self.model.device)

        outputs = self.model(full_ids)
        logits = outputs.logits  # (1, T, vocab)

        shift_logits = logits[:, :-1, :]
        shift_labels = full_ids[:, 1:]

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

        action_start = prompt_ids.shape[-1] - 1
        action_log_probs = log_probs[:, action_start:, :]
        action_labels = shift_labels[:, action_start:]

        token_log_probs = action_log_probs.gather(
            2, action_labels.unsqueeze(-1)
        ).squeeze(-1)

        return token_log_probs.sum()

    def save(self, path: str):
        from pathlib import Path
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path: str):
        from pathlib import Path
        import torch
        if not Path(path).exists():
            raise FileNotFoundError(f"No checkpoint at {path}")
        self.model.load_state_dict(
            torch.load(f"{path}/pytorch_model.bin", map_location=self.model.device),
            strict=False,
        )

    def _format_prompt(self, messages: list[dict]) -> str:
        """
        Convert message list to a single string the model can condition on.

        Format:
            <|system|>…</s>
            <|user|>…</s>
            <|assistant|>
        """
        parts = [f"<|system|>\n{SYSTEM_PROMPT}\n</s>"]
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|{role}|>\n{content}\n</s>")
        parts.append("<|assistant|>\n")
        return "\n".join(parts)
