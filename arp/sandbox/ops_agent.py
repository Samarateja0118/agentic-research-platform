"""CI-failure triage agent: reads a failing CI log, investigates in a sandbox, proposes a patch.

Guardrails:
- tool allowlist: list_files, read_file, search, replace_in_file, run_tests, finish. No shell.
- tests/ is read-only, edits must match exactly once, paths cannot leave the workspace.
- step budget (model calls) and a sandbox wall clock per test run.
- nothing leaves the machine without a human: opening a PR goes through the MCP server, which
  refuses until ``arp approve <id>`` is run for that exact patch.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..agents.loop import parse_json
from ..llm import LLM, cost_usd
from ..obs.trace import TracedLLM, Tracer
from .docker_sandbox import DockerSandbox, SandboxViolation, Workspace

PROJECT = Path("evals/bugs/project")
BUGS = Path("evals/bugs/bugs.json")

SYSTEM = """You are a CI triage engineer. A test run failed. Find the bug in the source code and fix it with the smallest change.

Rules:
- Tests are correct and read-only. Fix the code under ledger/, never the tests.
- Read the failing test and the code it calls before editing.
- Edit with replace_in_file: `old` must match the file exactly once (copy it from read_file output).
- After editing, run_tests to confirm. When the full suite passes, call finish with a one-line root cause.
"""

TOOLS = [
    {
        "name": "list_files",
        "description": "List files in the repository.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "read_file",
        "description": "Read a file with line numbers.",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    },
    {
        "name": "search",
        "description": "Regex search over all files; returns matching lines.",
        "input_schema": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
        },
    },
    {
        "name": "replace_in_file",
        "description": "Replace exact text `old` with `new` in a file. `old` must occur exactly once.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}},
            "required": ["path", "old", "new"],
        },
    },
    {
        "name": "run_tests",
        "description": "Run pytest in the sandbox. Optional target, e.g. tests/test_money.py.",
        "input_schema": {"type": "object", "properties": {"target": {"type": "string"}}},
    },
    {
        "name": "finish",
        "description": "Stop and report the root cause of the failure.",
        "input_schema": {
            "type": "object",
            "properties": {"root_cause": {"type": "string"}},
            "required": ["root_cause"],
        },
    },
]
ALLOWED = {t["name"] for t in TOOLS}


@dataclass
class TriageResult:
    bug: str
    mode: str
    fixed: bool
    diff: str
    root_cause: str
    steps: int
    tool_calls: int
    violations: list[str] = field(default_factory=list)
    hit_budget: bool = False
    tests_after: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    wall_s: float = 0.0
    patched: dict[str, str] = field(default_factory=dict, repr=False)


def apply_bug(ws: Workspace, bug: dict) -> None:
    text = ws.read(bug["file"])
    for e in bug["edits"]:
        assert text.count(e["old"]) == 1, f"bug {bug['id']} edit does not apply"
        text = text.replace(e["old"], e["new"])
    ws.resolve(bug["file"]).write_text(text)
    ws.original[bug["file"]] = text  # the buggy version is the baseline the agent's diff is measured from


def ci_log(output: str, limit: int = 3000) -> str:
    return output if len(output) <= limit else "...\n" + output[-limit:]


class OpsAgent:
    def __init__(self, llm: LLM, tracer: Tracer, sandbox: DockerSandbox, max_steps: int = 12):
        self.llm, self.tracer, self.sb, self.max_steps = (
            TracedLLM(llm, tracer, "ops"),
            tracer,
            sandbox,
            max_steps,
        )
        self.ws = sandbox.ws
        self.violations: list[str] = []
        self.tool_calls = 0

    async def tool(self, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        self.tool_calls += 1
        with self.tracer.span("tool", name, agent="ops", args=args) as sp:
            try:
                if name not in ALLOWED:
                    raise SandboxViolation(f"tool {name} is not allowed")
                out = await self._dispatch(name, args)
                sp.set(result=out[:2000], is_error=False)
                return out, False
            except (SandboxViolation, FileNotFoundError, IsADirectoryError, ValueError, KeyError) as exc:
                if isinstance(exc, SandboxViolation):
                    self.violations.append(str(exc))
                sp.set(result=str(exc), is_error=True)
                return f"error: {exc}", True

    async def _dispatch(self, name: str, a: dict[str, Any]) -> str:
        if name == "list_files":
            return "\n".join(self.ws.files())
        if name == "read_file":
            lines = self.ws.read(a["path"]).splitlines()
            return "\n".join(f"{i + 1:4d}| {ln}" for i, ln in enumerate(lines))[:6000]
        if name == "search":
            rx = re.compile(a["pattern"])
            hits = [
                f"{rel}:{i + 1}: {ln}"
                for rel in self.ws.files()
                if rel.endswith((".py", ".ini", ".txt"))
                for i, ln in enumerate(self.ws.read(rel).splitlines())
                if rx.search(ln)
            ]
            return "\n".join(hits[:60]) or "no matches"
        if name == "replace_in_file":
            text = self.ws.read(a["path"])
            n = text.count(a["old"])
            if n != 1:
                raise ValueError(f"`old` occurs {n} times in {a['path']}; it must occur exactly once")
            self.ws.write(a["path"], text.replace(a["old"], a["new"]))
            return f"edited {a['path']}"
        if name == "run_tests":
            r = await self.sb.run_tests(a.get("target", ""))
            return f"exit code {r.exit_code}\n{ci_log(r.output, 2500)}"
        if name == "finish":
            return "ok"
        raise SandboxViolation(name)

    async def run(self, log: str) -> tuple[str, int, bool]:
        messages: list[dict] = [{"role": "user", "content": f"CI failed. Log:\n```\n{log}\n```"}]
        for step in range(self.max_steps):
            resp = await self.llm.chat(SYSTEM, messages, TOOLS, max_tokens=700)
            messages.append(resp.as_message())
            if not resp.tool_calls:
                return resp.text, step + 1, False
            for call in resp.tool_calls:
                out, is_err = await self.tool(call.name, call.args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": out,
                        "is_error": is_err,
                    }
                )
                if call.name == "finish":
                    return call.args.get("root_cause", ""), step + 1, False
        return "step budget exhausted", self.max_steps, True


async def oneshot(llm: LLM, tracer: Tracer, ws: Workspace, log: str) -> tuple[str, list[str]]:
    """Baseline: one model call that sees the log and every source file, returns one edit."""
    src = "\n\n".join(f"### {rel}\n{ws.read(rel)}" for rel in ws.files() if rel.startswith("ledger/"))
    user = (
        f"CI failed. Log:\n```\n{log}\n```\n\nSource files:\n{src}\n\nReply with JSON only: "
        '{"path": "...", "old": "exact text to replace", "new": "replacement", "root_cause": "..."}'
    )
    resp = await TracedLLM(llm, tracer, "oneshot").chat(
        "Fix the bug with the smallest edit. Tests are correct.",
        [{"role": "user", "content": user}],
        max_tokens=1200,
        json_mode=True,
    )
    data = parse_json(resp.text) or {}
    violations = []
    try:
        text = ws.read(data["path"])
        if text.count(data["old"]) != 1:
            raise ValueError("old text not found exactly once")
        ws.write(data["path"], text.replace(data["old"], data["new"]))
    except (KeyError, ValueError, SandboxViolation, FileNotFoundError, TypeError) as exc:
        violations.append(f"edit not applied: {exc}")
    return str(data.get("root_cause", "")), violations


async def triage_bug(bug: dict, llm: LLM, mode: str, trace_path: Path) -> TriageResult:
    t0 = time.time()
    tracer = Tracer(trace_path, run_id=f"{mode}-{bug['id']}")
    ws = Workspace(PROJECT)
    try:
        apply_bug(ws, bug)
        sb = DockerSandbox(ws)
        before = await sb.run_tests()
        log = ci_log(before.output)
        with tracer.span("run", bug["id"], mode=mode):
            if mode == "agent":
                agent = OpsAgent(llm, tracer, sb)
                root, steps, hit = await agent.run(log)
                violations, calls = agent.violations, agent.tool_calls
            else:
                root, violations = await oneshot(llm, tracer, ws, log)
                steps, hit, calls = 1, False, 0
        after = await sb.run_tests()  # verification is ours, not the agent's claim
        totals = tracer.totals()
        return TriageResult(
            bug["id"],
            mode,
            after.exit_code == 0 and bool(ws.diff()),
            ws.diff(),
            root,
            steps,
            calls,
            violations,
            hit,
            after.output.strip().splitlines()[-1] if after.output.strip() else "",
            totals["input_tokens"],
            totals["output_tokens"],
            round(time.time() - t0, 1),
            ws.changed_files(),
        )
    finally:
        ws.cleanup()


async def triage_cli(a) -> int:
    from ..llm import make_llm

    bugs = json.loads(BUGS.read_text())
    if not a.all:
        bugs = [b for b in bugs if b["id"] == a.bug]
    llm = make_llm(a.llm)
    out = Path(a.out or f"evals/results/ops-{a.mode}.json")
    rows = []
    for bug in bugs:
        r = await triage_bug(bug, llm, a.mode, Path("evals/traces") / out.stem / f"{bug['id']}.jsonl")
        row = {k: v for k, v in r.__dict__.items() if k != "patched"} | {
            "kind": bug["kind"],
            "cost_at_claude-haiku-5-5": round(
                cost_usd("claude-haiku-5-5", r.input_tokens, r.output_tokens), 6
            ),
            "cost_at_claude-sonnet-5-5": round(
                cost_usd("claude-sonnet-5-5", r.input_tokens, r.output_tokens), 6
            ),
        }
        rows.append(row)
        print(
            f"[{a.mode}] {bug['id']} fixed={r.fixed} steps={r.steps} tools={r.tool_calls} "
            f"violations={len(r.violations)} {r.wall_s}s :: {r.root_cause[:100]}",
            flush=True,
        )
        if a.open_pr and r.fixed:
            print(await request_pr(bug, r))
    n = len(rows) or 1
    summary = {
        "n": len(rows),
        "fixed": sum(r["fixed"] for r in rows),
        "fix_rate": round(sum(r["fixed"] for r in rows) / n, 3),
        "mean_steps": round(sum(r["steps"] for r in rows) / n, 1),
        "violations": sum(len(r["violations"]) for r in rows),
        "mean_input_tokens": round(sum(r["input_tokens"] for r in rows) / n),
        "mean_wall_s": round(sum(r["wall_s"] for r in rows) / n, 1),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"meta": {"mode": a.mode, "model": llm.model}, "summary": summary, "rows": rows}, indent=1)
    )
    print(json.dumps(summary, indent=1))
    return 0


async def request_pr(bug: dict, r: TriageResult) -> str:
    """Ask the MCP server (as principal 'ops') to open a draft PR. Without approval it refuses and
    files an approval request; nothing is pushed until a human approves that exact payload."""
    from ..agents.toolbox import MCPToolbox

    async with MCPToolbox("ops") as tb:
        text, _ = await tb.call(
            "ops",
            "github_create_pull_request",
            {
                "owner": "Samarateja0118",
                "repo": "agentic-research-platform",
                "base": "main",
                "branch": f"ops/fix-{bug['id']}",
                "title": f"fix({bug['id']}): {r.root_cause[:60]}",
                "body": f"Proposed by the ops agent for seeded bug {bug['id']}.\n\n```diff\n{r.diff}\n```",
                "files": {f"{PROJECT}/{k}": v for k, v in r.patched.items()},
            },
        )
        return text
