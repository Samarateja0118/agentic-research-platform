"""Re-run every recorded trace in a results file through the current agent code."""

from __future__ import annotations

import json
from pathlib import Path

from ..agents.research import MultiAgentResearcher, SingleAgentResearcher
from ..obs.replay import Divergence, ReplayLLM, ReplayToolbox
from ..obs.trace import Tracer, load_trace


async def replay_one(system: str, trace_path: str, strict: bool = True) -> tuple[bool, str]:
    spans = load_trace(trace_path)
    run = next(s for s in spans if s["kind"] == "run")
    tools = run["attrs"].get("tools") or []
    llm, tb = ReplayLLM(spans, strict), ReplayToolbox(spans, tools)
    tracer = Tracer(None)
    try:
        if system == "multi_agent":
            res = await MultiAgentResearcher(llm, tracer, tb, tb).run(run["attrs"]["question"])
        else:
            res = await SingleAgentResearcher(llm, tracer, tb).run(run["attrs"]["question"])
    except Divergence as exc:
        return False, f"diverged: {exc}"
    if res.answer != (run["attrs"].get("answer") or ""):
        return False, "replayed answer differs from the recorded answer"
    extra = f" ({len(llm.divergences)} non-strict request mismatches)" if llm.divergences else ""
    return True, f"identical{extra}"


async def replay_traces(results: str, strict: bool = False) -> tuple[bool, str]:
    doc = json.loads(Path(results).read_text())
    system = doc["meta"]["system"]
    lines, ok_all = [], True
    for r in doc["rows"]:
        if r.get("error"):
            continue
        ok, msg = await replay_one(system, r["trace"], strict)
        ok_all &= ok
        lines.append(f"{r['id']}: {msg}")
    lines.append(
        f"replay {'OK' if ok_all else 'FAILED'}: {sum(1 for ln in lines if 'identical' in ln)}/{len(lines)} identical"
    )
    return ok_all, "\n".join(lines)
