"""Deterministic replay of a recorded trace.

Replaying re-runs the real orchestration code, but every model call and tool call is answered
from the trace. Uses:
- debug a bad run step by step without paying for model calls;
- regression-test orchestration code: if a code change alters what an agent sends, the
  request hash no longer matches the recording and replay reports the first divergence.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from ..llm import LLMResponse, ToolCall


class Divergence(Exception):
    pass


class ReplayLLM:
    def __init__(self, spans: list[dict], strict: bool = True):
        self.model = "replay"
        self.strict = strict
        self.by_agent: dict[str, dict[int, dict]] = defaultdict(dict)
        for s in spans:
            if s["kind"] == "llm":
                self.by_agent[s["attrs"]["agent"]][s["attrs"]["index"]] = s["attrs"]
        self.divergences: list[str] = []

    def replay(self, agent: str, index: int, request_hash: str) -> LLMResponse:
        rec = self.by_agent.get(agent, {}).get(index)
        if rec is None:
            raise Divergence(f"{agent}#{index}: no recorded call (the new code makes more calls)")
        if rec["request_hash"] != request_hash:
            msg = f"{agent}#{index}: request differs from recording ({rec['request_hash']} vs {request_hash})"
            if self.strict:
                raise Divergence(msg)
            self.divergences.append(msg)
        r = rec["response"]
        return LLMResponse(
            r["text"],
            [ToolCall(**c) for c in r["tool_calls"]],
            rec.get("input_tokens", 0),
            rec.get("output_tokens", 0),
            rec.get("model", ""),
            r.get("stop_reason", ""),
            r.get("raw"),
        )

    async def chat(self, *a: Any, **k: Any) -> LLMResponse:  # pragma: no cover - TracedLLM calls replay()
        raise Divergence("ReplayLLM must be wrapped in TracedLLM")


class ReplayToolbox:
    def __init__(self, spans: list[dict], tools: list[dict]):
        self.tools = tools
        self.calls: dict[tuple, list[dict]] = defaultdict(list)
        for s in spans:
            if s["kind"] == "tool":
                a = s["attrs"]
                self.calls[(a["agent"], s["name"], json.dumps(a["args"], sort_keys=True))].append(a)

    def subset(self, names: list[str]) -> list[dict]:
        return [t for t in self.tools if t["name"] in names]

    async def call(self, agent: str, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        key = (agent, name, json.dumps(args, sort_keys=True))
        if not self.calls.get(key):
            raise Divergence(f"{agent}: tool call {name}({args}) was not in the recording")
        rec = self.calls[key].pop(0)
        return rec["result"], rec["is_error"]
