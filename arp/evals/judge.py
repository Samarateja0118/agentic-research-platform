"""LLM-as-judge: grades an answer against the computed reference answer on a 1-5 rubric."""

from __future__ import annotations

from ..agents.loop import parse_json
from ..agents.research import load_prompt
from ..llm import LLM


async def judge(llm: LLM, question: str, reference: str, answer: str) -> dict:
    user = (
        f"Question: {question}\n\nReference answer: {reference}\n\nAssistant's answer:\n{answer or '(empty)'}"
    )
    resp = await llm.chat(
        load_prompt("judge"), [{"role": "user", "content": user}], max_tokens=150, json_mode=True
    )
    data = parse_json(resp.text) or {}
    try:
        s = int(data.get("score", 0))
    except (TypeError, ValueError):
        s = 0
    return {
        "score": s if 1 <= s <= 5 else None,
        "reason": str(data.get("reason", ""))[:300],
        "judge_tokens": resp.input_tokens + resp.output_tokens,
    }
