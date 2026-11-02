"""Provider-neutral chat interface with tool calls.

Messages use one neutral shape so agents never depend on a provider:
  {"role": "user", "content": str}
  {"role": "assistant", "content": str, "tool_calls": [{"id", "name", "args"}], "_raw": provider blocks}
  {"role": "tool", "tool_call_id": str, "name": str, "content": str, "is_error": bool}

Backends: Anthropic (Claude API), any OpenAI-compatible server (llama.cpp, vLLM), and a
scripted client for tests. Replay of recorded traces lives in ``arp.obs.replay``.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

# USD per million tokens (Claude API list prices, 2026-10). Local models cost $0 in cash; the
# eval report also prices their token counts at these rates as a rough "if this ran on Claude" figure.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = PRICES.get(model, (0.0, 0.0))
    return (input_tokens * pin + output_tokens * pout) / 1e6


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    stop_reason: str = ""
    raw: Any = None
    latency_ms: float = 0.0

    def as_message(self) -> dict[str, Any]:
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": self.text,
            "tool_calls": [{"id": c.id, "name": c.name, "args": c.args} for c in self.tool_calls],
        }
        if self.raw is not None:
            msg["_raw"] = self.raw
        return msg


class LLM(Protocol):
    model: str

    async def chat(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> LLMResponse: ...


class OpenAICompatLLM:
    """llama.cpp ``llama-server --jinja`` or any /v1/chat/completions endpoint."""

    def __init__(self, base_url: str | None = None, model: str = "local", timeout: float = 600.0):
        self.base_url = (base_url or os.environ.get("ARP_LLM_BASE_URL", "http://localhost:8080/v1")).rstrip(
            "/"
        )
        self.model = model
        self._http = httpx.AsyncClient(timeout=timeout)

    @staticmethod
    def _messages(system: str, messages: list[dict]) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "assistant":
                o: dict[str, Any] = {"role": "assistant", "content": m.get("content") or ""}
                if m.get("tool_calls"):
                    o["tool_calls"] = [
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {"name": c["name"], "arguments": json.dumps(c["args"])},
                        }
                        for c in m["tool_calls"]
                    ]
                out.append(o)
            elif m["role"] == "tool":
                content = m["content"] if not m.get("is_error") else f"ERROR: {m['content']}"
                out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": content})
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out

    async def chat(self, system, messages, tools=None, max_tokens=1024, json_mode=False):
        body: dict[str, Any] = {
            "model": self.model,
            "messages": self._messages(system, messages),
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed": 7,
        }
        if tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["input_schema"],
                    },
                }
                for t in tools
            ]
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        start = time.perf_counter()
        resp = await self._http.post(f"{self.base_url}/chat/completions", json=body)
        resp.raise_for_status()
        d = resp.json()
        choice = d["choices"][0]
        msg = choice["message"]
        calls = []
        for tc in msg.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"_unparsed": tc["function"].get("arguments")}
            calls.append(ToolCall(tc.get("id") or uuid.uuid4().hex[:8], tc["function"]["name"], args))
        usage = d.get("usage") or {}
        return LLMResponse(
            msg.get("content") or "",
            calls,
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            d.get("model", self.model),
            choice.get("finish_reason", ""),
            None,
            (time.perf_counter() - start) * 1e3,
        )


class AnthropicLLM:
    """Claude via the official SDK. Manual loop (not the beta tool runner) because every call
    must pass through the tracer and the provider-neutral message format."""

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium"):
        import anthropic

        self.model = model
        self.effort = effort
        self._client = anthropic.AsyncAnthropic(max_retries=4)

    @staticmethod
    def _messages(messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            if m["role"] == "assistant":
                if m.get("_raw") is not None:
                    content = m["_raw"]  # pass thinking + tool_use blocks back unchanged
                else:
                    content = [{"type": "text", "text": m["content"]}] if m.get("content") else []
                    content += [
                        {"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["args"]}
                        for c in m.get("tool_calls") or []
                    ]
                out.append({"role": "assistant", "content": content})
            elif m["role"] == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": m["tool_call_id"],
                    "content": m["content"],
                    "is_error": bool(m.get("is_error")),
                }
                # all results for one assistant turn go in a single user message
                if (
                    out
                    and out[-1]["role"] == "user"
                    and isinstance(out[-1]["content"], list)
                    and out[-1]["content"]
                    and out[-1]["content"][0].get("type") == "tool_result"
                ):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
            else:
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append({"type": "text", "text": m["content"]})
                else:
                    out.append({"role": "user", "content": m["content"]})
        return out

    async def chat(self, system, messages, tools=None, max_tokens=4096, json_mode=False):
        import anthropic

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max(max_tokens, 2048),
            "system": system,
            "messages": self._messages(messages),
            "output_config": {"effort": self.effort},
        }
        if tools:
            kwargs["tools"] = [
                {"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]}
                for t in tools
            ]
        start = time.perf_counter()
        try:
            resp = await self._client.messages.create(**kwargs)
        except anthropic.RateLimitError:
            raise
        except anthropic.APIStatusError as exc:
            raise RuntimeError(f"Claude API error {exc.status_code}: {exc.message}") from exc
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in resp.content if b.type == "tool_use"]
        raw = [b.model_dump(exclude_none=True) for b in resp.content]
        if resp.stop_reason == "refusal":
            text = text or "[refused]"
        return LLMResponse(
            text,
            calls,
            resp.usage.input_tokens,
            resp.usage.output_tokens,
            self.model,
            resp.stop_reason or "",
            raw,
            (time.perf_counter() - start) * 1e3,
        )


class ScriptedLLM:
    """Deterministic test double: returns queued responses per call."""

    def __init__(self, responses: list[LLMResponse | str], model: str = "scripted"):
        self.responses = list(responses)
        self.model = model
        self.calls: list[dict] = []

    async def chat(self, system, messages, tools=None, max_tokens=1024, json_mode=False):
        self.calls.append({"system": system, "messages": [dict(m) for m in messages], "tools": tools})
        if not self.responses:
            raise RuntimeError("ScriptedLLM ran out of responses")
        r = self.responses.pop(0)
        return LLMResponse(r, model=self.model) if isinstance(r, str) else r


def make_llm(spec: str) -> LLM:
    """``local``, ``local:<base_url>``, or ``claude:<model>[@effort]``."""
    if spec.startswith("claude"):
        model, _, effort = (spec.split(":", 1)[1] if ":" in spec else "claude-opus-5-5").partition("@")
        return AnthropicLLM(model, effort or "medium")
    if spec.startswith("local"):
        base = spec.split(":", 1)[1] if ":" in spec and "//" in spec else None
        return OpenAICompatLLM(base, model=os.environ.get("ARP_LOCAL_MODEL", "qwen3-4b-instruct-2507-q4_k_m"))
    raise ValueError(f"unknown llm spec {spec}")
