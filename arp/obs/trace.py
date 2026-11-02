"""Trace recording: nested spans written as JSONL, one file per run.

Span kinds: ``run`` (one question), ``agent`` (planner, worker[i], critic...), ``llm`` (one model
call, with the full request and response so it can be replayed), ``tool`` (one MCP call).
Parents come from a contextvar, so spans opened inside ``asyncio.gather`` workers nest correctly.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..llm import LLM, LLMResponse, cost_usd

_current: contextvars.ContextVar[Span | None] = contextvars.ContextVar("arp_span", default=None)


def request_hash(system: str, messages: list[dict], tools: list[dict] | None) -> str:
    clean = [{k: v for k, v in m.items() if k != "_raw"} for m in messages]
    blob = json.dumps(
        {"s": system, "m": clean, "t": [t["name"] for t in tools or []]}, sort_keys=True, default=str
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass
class Span:
    kind: str
    name: str
    span_id: str
    parent_id: str | None
    start: float
    attrs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def set(self, **kv: Any) -> None:
        self.attrs.update(kv)


class Tracer:
    def __init__(self, path: str | Path | None = None, run_id: str | None = None):
        self.run_id = run_id or uuid.uuid4().hex[:10]
        self.path = Path(path) if path else None
        self.spans: list[dict] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("")

    @contextmanager
    def span(self, kind: str, name: str, **attrs: Any):
        parent = _current.get()
        sp = Span(
            kind, name, uuid.uuid4().hex[:12], parent.span_id if parent else None, time.time(), dict(attrs)
        )
        token = _current.set(sp)
        try:
            yield sp
        except BaseException as exc:
            sp.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            _current.reset(token)
            self._emit(sp)

    def _emit(self, sp: Span) -> None:
        rec = {
            "run_id": self.run_id,
            "span_id": sp.span_id,
            "parent_id": sp.parent_id,
            "kind": sp.kind,
            "name": sp.name,
            "start": round(sp.start, 4),
            "dur_ms": round((time.time() - sp.start) * 1e3, 1),
            "attrs": sp.attrs,
            "error": sp.error,
        }
        self.spans.append(rec)
        if self.path:
            with self.path.open("a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")

    # aggregate helpers used by the eval runner
    def totals(self) -> dict[str, Any]:
        llm = [s for s in self.spans if s["kind"] == "llm"]
        tools = [s for s in self.spans if s["kind"] == "tool"]
        return {
            "llm_calls": len(llm),
            "tool_calls": len(tools),
            "tool_errors": sum(1 for s in tools if s["attrs"].get("is_error")),
            "input_tokens": sum(s["attrs"].get("input_tokens", 0) for s in llm),
            "output_tokens": sum(s["attrs"].get("output_tokens", 0) for s in llm),
            "cost_usd": round(sum(s["attrs"].get("cost_usd", 0.0) for s in llm), 6),
            "llm_ms": round(sum(s["dur_ms"] for s in llm), 1),
        }


def load_trace(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


class TracedLLM:
    """Wraps an LLM so each call is a span carrying the full request and response."""

    def __init__(self, inner: LLM, tracer: Tracer, agent: str):
        self.inner = inner
        self.tracer = tracer
        self.agent = agent
        self.model = inner.model
        self.n = 0

    async def chat(self, system, messages, tools=None, max_tokens=1024, json_mode=False) -> LLMResponse:
        idx = self.n
        self.n += 1
        with self.tracer.span(
            "llm",
            f"{self.agent}#{idx}",
            agent=self.agent,
            index=idx,
            model=self.inner.model,
            request_hash=request_hash(system, messages, tools),
        ) as sp:  # noqa: E501
            sp.set(
                request={
                    "system": system,
                    "messages": [{k: v for k, v in m.items() if k != "_raw"} for m in messages],
                    "tools": [t["name"] for t in tools or []],
                    "max_tokens": max_tokens,
                }
            )
            rhash = request_hash(system, messages, tools)
            if hasattr(self.inner, "replay"):
                resp = self.inner.replay(self.agent, idx, rhash)
                sp.set(replayed=True)
            else:
                resp = await self.inner.chat(system, messages, tools, max_tokens, json_mode)
            sp.set(
                response={
                    "text": resp.text,
                    "tool_calls": [c.__dict__ for c in resp.tool_calls],
                    "stop_reason": resp.stop_reason,
                    "raw": resp.raw,
                },
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                cost_usd=cost_usd(resp.model or self.inner.model, resp.input_tokens, resp.output_tokens),
            )
            return resp
