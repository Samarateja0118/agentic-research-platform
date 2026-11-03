"""Render a trace file as an indented span tree with timings, tokens and tool outcomes."""

from __future__ import annotations

from collections import defaultdict

from .trace import load_trace


def render(path: str, show_io: bool = False) -> str:
    spans = load_trace(path)
    kids: dict[str | None, list[dict]] = defaultdict(list)
    for s in spans:
        kids[s["parent_id"]].append(s)
    for v in kids.values():
        v.sort(key=lambda s: s["start"])
    t0 = min((s["start"] for s in spans), default=0)
    out: list[str] = []

    def line(s: dict, depth: int) -> None:
        a = s["attrs"]
        extra = ""
        if s["kind"] == "llm":
            calls = [c["name"] for c in a.get("response", {}).get("tool_calls", [])]
            extra = f" in={a.get('input_tokens')} out={a.get('output_tokens')}" + (
                f" -> {calls}" if calls else ""
            )
        elif s["kind"] == "tool":
            extra = f" {a.get('args')} {'ERROR' if a.get('is_error') else 'ok'}"
        elif s["kind"] == "agent" and a.get("verdict"):
            extra = f" verdict={a['verdict']}"
        err = f"  !! {s['error']}" if s.get("error") else ""
        out.append(
            f"{'  ' * depth}+{s['start'] - t0:7.1f}s {s['dur_ms'] / 1e3:6.1f}s  {s['kind']:<5} {s['name']}{extra}{err}"
        )
        if show_io and s["kind"] == "llm":
            out.append(f"{'  ' * (depth + 1)}  >> {a.get('response', {}).get('text', '')[:300]!r}")
        if show_io and s["kind"] == "tool":
            out.append(f"{'  ' * (depth + 1)}  << {str(a.get('result', ''))[:300]!r}")
        for k in kids.get(s["span_id"], []):
            line(k, depth + 1)

    for root in kids.get(None, []):
        line(root, 0)
    return "\n".join(out)
