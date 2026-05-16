"""
LLM Client — Groq backend with Ollama fallback path.
====================================================
Wraps Groq's API for the Llama 3.3 70B model. Same pattern as AP project
for consistency. Records latency and tokens for Langfuse spans.
"""
import os
import time
from dataclasses import dataclass
from typing import Optional, Any
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    model: str
    raw: Any = None


class LLMClient:
    """Wraps Groq (free tier) with a clean API."""

    def __init__(self, model: Optional[str] = None, temperature: float = 0.2):
        self.model = model or os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        self.temperature = temperature
        self.backend = "groq"
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set. Get one free at https://console.groq.com")
        try:
            from groq import Groq
        except ImportError:
            raise RuntimeError("groq package not installed. Run: pip install groq")
        self._client = Groq(api_key=api_key)
        logger.info(f"LLMClient: backend={self.backend} model={self.model}")

    def complete(self, system: str, user: str,
                 max_tokens: int = 1024,
                 tools: Optional[list[dict]] = None,
                 tool_choice: Optional[str] = None) -> LLMResponse:
        """
        Standard chat completion. If `tools` is provided, supports function calling.
        Returns LLMResponse with .text and optional .raw.tool_calls.
        """
        t0 = time.time()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
        }
        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice

        resp = self._client.chat.completions.create(**kwargs)
        latency_ms = int((time.time() - t0) * 1000)

        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = resp.usage
        return LLMResponse(
            text=text,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency_ms,
            model=self.model,
            raw=choice.message,
        )
