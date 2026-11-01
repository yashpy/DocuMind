"""Provider-agnostic LLM layer (OpenAI / Groq / offline deterministic stub).

The offline stub is extractive and grounded — it answers only from the provided context —
so the full agent graph and the eval harness run deterministically in CI with no API key.
Real providers plug in behind the same `generate` signature.
"""

from __future__ import annotations

import os
import re
from typing import List, Protocol


class LLM(Protocol):
    def generate(self, system: str, user: str) -> str: ...


class StubLLM:
    """Deterministic extractive LLM used for offline runs/tests.

    Pulls the sentences from CONTEXT most relevant to QUESTION. Multi-part questions pull
    one supporting sentence per distinct intent, which exercises the agent's decomposition.
    """

    def generate(self, system: str, user: str) -> str:
        context = self._section(user, "CONTEXT")
        question = self._section(user, "QUESTION")
        sentences = [
            s.lstrip("- ").strip()
            for s in re.split(r"(?<=[.!?])\s+", context)
            if s.strip()
        ]
        if not sentences:
            return "I don't have enough information to answer that."

        q_tokens = set(re.findall(r"[a-z0-9]+", question.lower()))
        scored = sorted(
            sentences,
            key=lambda s: len(q_tokens & set(re.findall(r"[a-z0-9]+", s.lower()))),
            reverse=True,
        )
        # Keep the strongest 1-2 supporting sentences with nonzero overlap.
        picked = [
            s for s in scored[:2]
            if q_tokens & set(re.findall(r"[a-z0-9]+", s.lower()))
        ]
        return " ".join(picked) if picked else "I don't have enough information to answer that."

    @staticmethod
    def _section(text: str, header: str) -> str:
        m = re.search(rf"{header}:\n(.*?)(?:\n[A-Z ]+:\n|\Z)", text, re.S)
        return m.group(1).strip() if m else ""


class _HTTPChatLLM:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url, self.api_key, self.model = base_url, api_key, model

    def generate(self, system: str, user: str) -> str:
        import requests  # lazy import by design

        resp = requests.post(
            self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.1,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def get_llm() -> LLM:
    name = os.environ.get("DOCUMIND_PROVIDER", "stub").lower()
    if name == "openai":
        return _HTTPChatLLM("https://api.openai.com/v1/chat/completions", os.environ["OPENAI_API_KEY"], "gpt-4o")
    if name == "groq":
        return _HTTPChatLLM("https://api.groq.com/openai/v1/chat/completions", os.environ["GROQ_API_KEY"], "llama-3.3-70b-versatile")
    return StubLLM()
