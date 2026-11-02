"""The tool-use loop every agent runs: call model -> run requested tools -> repeat, under a step budget."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from ..llm import LLM
from .toolbox import Toolbox


@dataclass
class LoopResult:
    text: str
    messages: list[dict]
    steps: int
    evidence: list[dict] = field(default_factory=list)  # {"tool", "args", "result", "is_error"}
    hit_budget: bool = False
    repeated_calls: int = 0


async def run_tool_loop(
    llm: LLM,
    system: str,
    user: str,
    toolbox: Toolbox | None,
    tools: list[dict],
    agent: str,
    max_steps: int = 6,
    max_tokens: int = 700,
) -> LoopResult:
    messages: list[dict] = [{"role": "user", "content": user}]
    evidence: list[dict] = []
    seen: dict[str, str] = {}
    repeated = 0
    for step in range(max_steps):
        resp = await llm.chat(system, messages, tools or None, max_tokens=max_tokens)
        messages.append(resp.as_message())
        if not resp.tool_calls or toolbox is None:
            return LoopResult(resp.text, messages, step + 1, evidence, False, repeated)

        async def run_one(call):
            key = f"{call.name}:{json.dumps(call.args, sort_keys=True)}"
            if key in seen:
                return (
                    call,
                    seen[key] + "\n(note: identical call already made; reuse this result)",
                    False,
                    True,
                )
            text, is_error = await toolbox.call(agent, call.name, call.args)
            seen[key] = text
            return call, text, is_error, False

        results = await asyncio.gather(*(run_one(c) for c in resp.tool_calls))
        for call, text, is_error, was_repeat in results:
            repeated += int(was_repeat)
            if not was_repeat:
                evidence.append({"tool": call.name, "args": call.args, "result": text, "is_error": is_error})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": text,
                    "is_error": is_error,
                }
            )
    # budget exhausted: one last call without tools forces an answer from what was gathered
    messages.append(
        {"role": "user", "content": "Step budget reached. Answer now from the tool results above."}
    )
    resp = await llm.chat(system, messages, None, max_tokens=max_tokens)
    messages.append(resp.as_message())
    return LoopResult(resp.text, messages, max_steps + 1, evidence, True, repeated)


def parse_json(text: str) -> dict[str, Any] | None:
    """Tolerant JSON extraction: models wrap JSON in prose or code fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("\n") + 1 :] if "\n" in text else text
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
