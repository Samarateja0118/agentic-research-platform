"""Command line: ``arp ask``, ``arp eval``, ``arp compare``, ``arp gate``, ``arp replay``, ``arp ops``, ``arp approve``."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


def _llm(spec: str):
    from .llm import make_llm

    return make_llm(spec)


def cmd_ask(a) -> int:
    from .agents.research import MultiAgentResearcher, SingleAgentResearcher
    from .agents.toolbox import MCPToolbox
    from .obs.trace import Tracer

    async def go():
        tracer = Tracer(a.trace)
        llm = _llm(a.llm)
        async with MCPToolbox("researcher", tracer) as tb, MCPToolbox("critic", tracer) as ctb:
            r = await (
                MultiAgentResearcher(llm, tracer, tb, ctb)
                if a.system == "multi_agent"
                else SingleAgentResearcher(llm, tracer, tb)
            ).run(a.question)
        print(r.answer)
        print(json.dumps(tracer.totals()), file=sys.stderr)
        print(f"trace: {a.trace}", file=sys.stderr)

    asyncio.run(go())
    return 0


def cmd_eval(a) -> int:
    from .evals.runner import run_eval

    judge = _llm(a.judge) if a.judge != "none" else None
    doc = asyncio.run(
        run_eval(
            a.system,
            _llm(a.llm),
            a.dataset,
            a.out,
            judge,
            a.ids.split(",") if a.ids else None,
            a.concurrency,
            Path(a.prompt_dir) if a.prompt_dir else None,
            a.label,
        )
    )
    print(json.dumps(doc["summary"], indent=1))
    return 0


def cmd_compare(a) -> int:
    from .evals.report import compare_markdown

    print(compare_markdown([json.loads(Path(p).read_text()) for p in a.results]))
    return 0


def cmd_gate(a) -> int:
    from .evals.gate import gate

    ok, report = gate(a.baseline, a.candidate, a.max_drop, a.prompts_dir)
    print(report)
    return 0 if ok else 1


def cmd_replay(a) -> int:
    from .evals.replay_check import replay_traces

    ok, report = asyncio.run(replay_traces(a.results, a.strict))
    print(report)
    return 0 if ok else 1


def cmd_rescore(a) -> int:
    from .evals.gate import rescore

    ok, report = rescore(a.results, a.dataset, a.write)
    print(report)
    return 0 if ok else 1


def cmd_trace(a) -> int:
    from .obs.view import render

    print(render(a.path, a.io))
    return 0


def cmd_approve(a) -> int:
    from .mcp_server.policy import ApprovalStore

    store = ApprovalStore(os.environ.get("ARP_APPROVALS", ".arp/approvals"))
    req = json.loads((store.root / f"{a.request_id}.request.json").read_text())
    print(f"Approval request {a.request_id}: {req['summary']}")
    for path, content in req["payload"].get("files", {}).items():
        print(f"--- {path} ({len(content)} bytes)")
    if not a.yes and input("approve? [y/N] ").strip().lower() != "y":
        print("not approved")
        return 1
    store.approve(a.request_id, a.approver)
    print("approved")
    return 0


def cmd_ops(a) -> int:
    from .sandbox.ops_agent import triage_cli

    return asyncio.run(triage_cli(a))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="arp")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ask", help="answer one question")
    s.add_argument("question")
    s.add_argument("--system", default="multi_agent", choices=["multi_agent", "single_agent"])
    s.add_argument("--llm", default=os.environ.get("ARP_LLM", "local"))
    s.add_argument("--trace", default=".arp/last-trace.jsonl")
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("eval", help="run the research eval set")
    s.add_argument("--system", default="multi_agent", choices=["multi_agent", "single_agent"])
    s.add_argument("--llm", default=os.environ.get("ARP_LLM", "local"))
    s.add_argument("--judge", default=os.environ.get("ARP_JUDGE", "local"))
    s.add_argument("--dataset", default="evals/datasets/research.jsonl")
    s.add_argument("--out", required=True)
    s.add_argument("--ids", default="")
    s.add_argument("--concurrency", type=int, default=1)
    s.add_argument("--prompt-dir", default="")
    s.add_argument("--label", default="")
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("compare", help="markdown comparison of result files")
    s.add_argument("results", nargs="+")
    s.set_defaults(fn=cmd_compare)

    s = sub.add_parser("gate", help="fail (exit 1) when a candidate regresses against the baseline")
    s.add_argument("--baseline", required=True)
    s.add_argument("--candidate", required=True)
    s.add_argument("--max-drop", type=float, default=0.05)
    s.add_argument("--prompts-dir", default="prompts")
    s.set_defaults(fn=cmd_gate)

    s = sub.add_parser("replay", help="re-run recorded traces through the current code")
    s.add_argument("results")
    s.add_argument("--strict", action="store_true")
    s.set_defaults(fn=cmd_replay)

    s = sub.add_parser("rescore", help="re-apply deterministic checks to recorded answers")
    s.add_argument("results")
    s.add_argument("--dataset", default="evals/datasets/research.jsonl")
    s.add_argument("--write", action="store_true", help="update pass/fail fields from the recorded answers")
    s.set_defaults(fn=cmd_rescore)

    s = sub.add_parser("trace", help="print a recorded trace as a span tree")
    s.add_argument("path")
    s.add_argument("--io", action="store_true", help="include model outputs and tool results")
    s.set_defaults(fn=cmd_trace)

    s = sub.add_parser("approve", help="human approval for a pending write (e.g. a pull request)")
    s.add_argument("request_id")
    s.add_argument("--approver", default=os.environ.get("USER", "human"))
    s.add_argument("--yes", action="store_true")
    s.set_defaults(fn=cmd_approve)

    s = sub.add_parser("ops", help="CI-failure triage agent in a Docker sandbox")
    s.add_argument("--bug", help="seeded bug id from evals/bugs/bugs.json")
    s.add_argument("--all", action="store_true", help="run every seeded bug")
    s.add_argument("--mode", default="agent", choices=["agent", "oneshot"])
    s.add_argument("--llm", default=os.environ.get("ARP_LLM", "local"))
    s.add_argument("--out", default="")
    s.add_argument("--open-pr", action="store_true", help="request approval and open a draft PR with the fix")
    s.set_defaults(fn=cmd_ops)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
