"""
/submission/model.py — submission contract.

The judge calls load_model() to obtain the hardened model.
Fill in the implementation below, then place this file at /submission/model.py.

Constraints
-----------
- load_model() must take no arguments.
- Return type must be (transformers.PreTrainedModel, PreTrainedTokenizer).
- The model must be loadable on the judge's hardware without additional setup.
"""
from typing import Tuple

from transformers import PreTrainedModel, PreTrainedTokenizerBase


def load_model() -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Load and return the hardened (model, tokenizer) pair.

    Example — loading a LoRA-merged checkpoint saved to ./checkpoints/hardened:

        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("./checkpoints/hardened")
        model = AutoModelForCausalLM.from_pretrained("./checkpoints/hardened")
        return model, tokenizer

    Example — loading a PEFT adapter on top of the base model:

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
        model = PeftModel.from_pretrained(base, "./checkpoints/adapter")
        tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
        return model, tokenizer
    """
    raise NotImplementedError(
        "Implement load_model() to return your hardened (model, tokenizer) tuple."
    )
