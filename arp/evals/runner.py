"""Run a research system over the eval set: trace every question, score it, judge it, cost it."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from ..agents.research import (
    MULTI_PROMPTS,
    SINGLE_PROMPTS,
    MultiAgentResearcher,
    SingleAgentResearcher,
    prompt_version,
    summarize_result,
)
from ..agents.toolbox import MCPToolbox
from ..llm import LLM, PRICES, cost_usd
from ..mcp_server.http import Cassette
from ..obs.trace import Tracer
from .checks import score
from .judge import judge

MAX_SUBTASKS = int(os.environ.get("ARP_MAX_SUBTASKS", "4"))
REF_PRICES = ["claude-haiku-5-5", "claude-sonnet-5-5"]


def load_dataset(path: str | Path, ids: list[str] | None = None) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return [r for r in rows if not ids or r["id"] in ids]


def file_digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


async def run_one(
    system: str, llm: LLM, q: dict, trace_path: Path, prompt_dir: Path | None
) -> dict[str, Any]:
    tracer = Tracer(trace_path, run_id=f"{system}-{q['id']}")
    t0 = time.time()
    error = None
    result = None
    with tracer.span("run", q["id"], system=system, question=q["question"]) as run_span:
        try:
            async with MCPToolbox("researcher", tracer) as tb:
                run_span.set(tools=tb.tools)
                if system == "multi_agent":
                    async with MCPToolbox("critic", tracer) as ctb:
                        result = await MultiAgentResearcher(
                            llm, tracer, tb, ctb, prompt_dir=prompt_dir, max_subtasks=MAX_SUBTASKS
                        ).run(q["question"])
                else:
                    result = await SingleAgentResearcher(llm, tracer, tb, prompt_dir=prompt_dir).run(
                        q["question"]
                    )
        except Exception as exc:  # a crashed run is a failed question, not a crashed eval
            error = f"{type(exc).__name__}: {exc}"
        run_span.set(answer=result.answer if result else None, error=error)
    answer = result.answer if result else ""
    row = {
        "id": q["id"],
        "category": q["category"],
        "hops": q["hops"],
        "system": system,
        "answer": answer,
        "error": error,
        "wall_s": round(time.time() - t0, 1),
        **score(q["checks"], answer),
        **tracer.totals(),
        "trace": str(trace_path),
    }
    for m in REF_PRICES:
        row[f"cost_at_{m}"] = round(cost_usd(m, row["input_tokens"], row["output_tokens"]), 6)
    if result:
        row["detail"] = summarize_result(result)
    return row


async def run_eval(
    system: str,
    llm: LLM,
    dataset: str,
    out: str | Path,
    judge_llm: LLM | None = None,
    ids: list[str] | None = None,
    concurrency: int = 1,
    prompt_dir: Path | None = None,
    label: str = "",
) -> dict[str, Any]:
    qs = load_dataset(dataset, ids)
    out = Path(out)
    name = out.stem
    trace_dir = Path("evals/traces") / name
    trace_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    rows: list[dict] = []
    if out.exists():  # resume an interrupted run if it used the same prompts and dataset
        prev = json.loads(out.read_text())
        names = MULTI_PROMPTS if system == "multi_agent" else SINGLE_PROMPTS
        if prev["meta"].get("prompt_version") == prompt_version(names, prompt_dir) and prev["meta"].get(
            "dataset_digest"
        ) == file_digest(dataset):
            rows = [r for r in prev["rows"] if not r.get("error")]
            done = {r["id"] for r in rows}
            qs = [q for q in qs if q["id"] not in done]
            print(f"[{name}] resuming: {len(done)} done, {len(qs)} to go", flush=True)

    async def go(q):
        async with sem:
            row = await run_one(system, llm, q, trace_dir / f"{q['id']}.jsonl", prompt_dir)
            if judge_llm is not None:
                row["judge"] = await judge(judge_llm, q["question"], q["reference"], row["answer"])
            rows.append(row)
            print(
                f"[{name}] {q['id']} pass={row['pass']} judge={row.get('judge', {}).get('score')} "
                f"llm_calls={row['llm_calls']} tools={row['tool_calls']} {row['wall_s']}s",
                flush=True,
            )
            _write(out, system, llm, dataset, prompt_dir, label, rows)

    await asyncio.gather(*(go(q) for q in qs))
    return _write(out, system, llm, dataset, prompt_dir, label, rows)


def _write(out: Path, system, llm, dataset, prompt_dir, label, rows) -> dict:
    rows = sorted(rows, key=lambda r: r["id"])
    names = MULTI_PROMPTS if system == "multi_agent" else SINGLE_PROMPTS
    cdir = Path(os.environ.get("ARP_CASSETTE_DIR", "evals/cassettes"))
    doc = {
        "meta": {
            "system": system,
            "model": llm.model,
            "label": label,
            "dataset": str(dataset),
            "dataset_digest": file_digest(dataset),
            "prompt_version": prompt_version(names, prompt_dir),
            "snapshot_digest": {p.stem: Cassette(p).digest() for p in sorted(cdir.glob("*.json"))},
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "prices_per_mtok": PRICES,
        },
        "summary": summarize(rows),
        "rows": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, default=str))
    return doc


def summarize(rows: list[dict]) -> dict[str, Any]:
    n = len(rows) or 1
    judged = [r["judge"]["score"] for r in rows if r.get("judge", {}).get("score")]
    s = {
        "n": len(rows),
        "pass_rate": round(sum(r["pass"] for r in rows) / n, 3),
        "check_fraction": round(sum(r["check_fraction"] for r in rows) / n, 3),
        "judge_mean": round(sum(judged) / len(judged), 2) if judged else None,
        "judge_ge4": round(sum(1 for j in judged if j >= 4) / len(judged), 3) if judged else None,
        "errors": sum(1 for r in rows if r["error"]),
        "mean_llm_calls": round(sum(r["llm_calls"] for r in rows) / n, 1),
        "mean_tool_calls": round(sum(r["tool_calls"] for r in rows) / n, 1),
        "tool_error_rate": round(
            sum(r["tool_errors"] for r in rows) / max(1, sum(r["tool_calls"] for r in rows)), 3
        ),
        "mean_input_tokens": round(sum(r["input_tokens"] for r in rows) / n),
        "mean_output_tokens": round(sum(r["output_tokens"] for r in rows) / n),
        "mean_wall_s": round(sum(r["wall_s"] for r in rows) / n, 1),
    }
    for m in REF_PRICES:
        s[f"mean_cost_at_{m}"] = round(sum(r[f"cost_at_{m}"] for r in rows) / n, 5)
    by_cat: dict[str, list] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r["pass"])
    s["pass_by_category"] = {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_cat.items())}
    return s
