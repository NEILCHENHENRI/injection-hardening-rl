"""
agent/api_agent.py — Anthropic API-backed agent.

Useful for:
  - Sanity-checking the environment pipeline without training anything
  - Generating initial trajectories to warm-start the local agent
  - Upper-bound performance comparison

Not trainable — the RL loop cannot update API model weights.
"""
from __future__ import annotations

from typing import Optional

from agent.base import BaseAgent, SYSTEM_PROMPT


class APIAgent(BaseAgent):
    """
    Agent backed by the Anthropic Messages API.

    Args:
        model        : Anthropic model string, e.g. "claude-sonnet-4-20250514"
        max_tokens   : token budget per response
        api_key      : if None, reads from ANTHROPIC_API_KEY env var
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 512,
        api_key: Optional[str] = None,
        max_history_turns: int = 20,
    ):
        super().__init__(max_history_turns=max_history_turns)
        self.model = model
        self.max_tokens = max_tokens

        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            )

    def _call_model(self, messages: list[dict]) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        return response.content[0].text
