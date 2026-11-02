"""Multi-agent researcher (planner -> parallel workers -> synthesizer -> critic) and the single-agent baseline."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..llm import LLM
from ..obs.trace import TracedLLM, Tracer
from .loop import parse_json, run_tool_loop
from .toolbox import Toolbox

PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"
CRITIC_TOOLS = ["pypi_package", "pypi_version", "pypi_releases", "github_repo", "github_read_file"]


def load_prompt(name: str, prompt_dir: Path | None = None) -> str:
    return ((prompt_dir or PROMPT_DIR) / f"{name}.md").read_text()


def prompt_version(names: list[str], prompt_dir: Path | None = None) -> str:
    h = hashlib.sha256()
    for n in sorted(names):
        h.update(n.encode())
        h.update(load_prompt(n, prompt_dir).encode())
    return h.hexdigest()[:12]


MULTI_PROMPTS = ["planner", "worker", "synthesizer", "critic"]
SINGLE_PROMPTS = ["single_agent"]


@dataclass
class ResearchResult:
    answer: str
    system: str
    plan: list[dict] = field(default_factory=list)
    critic: list[dict] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _tool_list(tools: list[dict]) -> str:
    return "\n".join(f"- {t['name']}: {t['description'].splitlines()[0]}" for t in tools)


def _evidence_digest(evidence: list[dict], budget: int = 6000) -> str:
    parts, used = [], 0
    for i, e in enumerate(evidence):
        chunk = (
            f"[E{i + 1}] {e['tool']}({e['args']}) -> {'ERROR ' if e['is_error'] else ''}{e['result'][:1200]}"
        )
        if used + len(chunk) > budget:
            parts.append(f"... {len(evidence) - i} more evidence items omitted")
            break
        parts.append(chunk)
        used += len(chunk)
    return "\n".join(parts) or "(no evidence collected)"


class MultiAgentResearcher:
    name = "multi_agent"

    def __init__(
        self,
        llm: LLM,
        tracer: Tracer,
        researcher_tools: Toolbox,
        critic_tools: Toolbox | None = None,
        max_subtasks: int = 4,
        worker_steps: int = 5,
        revisions: int = 1,
        worker_concurrency: int = 4,
        prompt_dir: Path | None = None,
    ):
        self.llm, self.tracer = llm, tracer
        self.tb, self.critic_tb = researcher_tools, critic_tools
        self.max_subtasks, self.worker_steps, self.revisions = max_subtasks, worker_steps, revisions
        self.sem = asyncio.Semaphore(worker_concurrency)
        self.p = {n: load_prompt(n, prompt_dir) for n in MULTI_PROMPTS}

    def _llm(self, agent: str) -> TracedLLM:
        return TracedLLM(self.llm, self.tracer, agent)

    async def plan(self, question: str) -> list[dict]:
        with self.tracer.span("agent", "planner", role="planner"):
            system = self.p["planner"].format(tool_list=_tool_list(self.tb.tools))
            resp = await self._llm("planner").chat(
                system, [{"role": "user", "content": question}], max_tokens=400, json_mode=True
            )
            data = parse_json(resp.text) or {}
            subtasks = [s for s in data.get("subtasks", []) if isinstance(s, dict) and s.get("question")]
            if not subtasks:  # planner failed: fall back to one worker on the whole question
                subtasks = [{"id": "t1", "question": question}]
            return subtasks[: self.max_subtasks]

    async def work(self, agent: str, task: str, question: str) -> dict:
        async with self.sem:
            with self.tracer.span("agent", agent, role="worker", task=task):
                user = f"Sub-question: {task}\n\n(Context, the overall question: {question})"
                res = await run_tool_loop(
                    self._llm(agent),
                    self.p["worker"],
                    user,
                    self.tb,
                    self.tb.tools,
                    agent,
                    max_steps=self.worker_steps,
                )
                return {
                    "agent": agent,
                    "task": task,
                    "findings": res.text,
                    "evidence": res.evidence,
                    "hit_budget": res.hit_budget,
                }

    async def synthesize(self, question: str, reports: list[dict], problems: list[str], agent: str) -> str:
        with self.tracer.span("agent", agent, role="synthesizer"):
            body = "\n\n".join(f"## {r['task']}\n{r['findings']}" for r in reports)
            user = f"Question: {question}\n\nFindings:\n{body}"
            if problems:
                user += (
                    "\n\nA verifier found these problems in your previous draft. Fix them:\n- "
                    + "\n- ".join(problems)
                )
            resp = await self._llm(agent).chat(
                self.p["synthesizer"], [{"role": "user", "content": user}], max_tokens=500
            )
            return resp.text

    async def verify(self, question: str, draft: str, evidence: list[dict], agent: str) -> dict:
        with self.tracer.span("agent", agent, role="critic") as sp:
            user = (
                f"Question: {question}\n\nDraft answer:\n{draft}\n\nEvidence (raw tool results):\n"
                f"{_evidence_digest(evidence)}"
            )
            tools = self.critic_tb.subset(CRITIC_TOOLS) if self.critic_tb else []
            res = await run_tool_loop(
                self._llm(agent),
                self.p["critic"],
                user,
                self.critic_tb,
                tools,
                agent,
                max_steps=3,
                max_tokens=400,
            )
            verdict = parse_json(res.text) or {
                "verdict": "pass",
                "problems": [],
                "followups": [],
                "parse_error": True,
            }
            verdict["evidence"] = res.evidence
            sp.set(verdict=verdict.get("verdict"), problems=verdict.get("problems"))
            return verdict

    async def run(self, question: str) -> ResearchResult:
        with self.tracer.span("agent", "multi_agent", role="orchestrator"):
            plan = await self.plan(question)
            reports = await asyncio.gather(
                *(self.work(f"worker[{i}]", s["question"], question) for i, s in enumerate(plan))
            )
            reports = list(reports)
            evidence = [e for r in reports for e in r["evidence"]]
            draft = await self.synthesize(question, reports, [], "synthesizer#0")
            critiques = []
            for rnd in range(self.revisions + 1):
                verdict = await self.verify(question, draft, evidence, f"critic#{rnd}")
                evidence += verdict.pop("evidence", [])
                critiques.append(verdict)
                if verdict.get("verdict") != "revise" or rnd == self.revisions:
                    break
                followups = [f for f in verdict.get("followups") or [] if isinstance(f, str)][:2]
                if followups:
                    extra = await asyncio.gather(
                        *(self.work(f"worker[r{rnd}.{j}]", f, question) for j, f in enumerate(followups))
                    )
                    reports += extra
                    evidence += [e for r in extra for e in r["evidence"]]
                draft = await self.synthesize(
                    question,
                    reports,
                    [str(p) for p in verdict.get("problems") or []],
                    f"synthesizer#{rnd + 1}",
                )
            return ResearchResult(draft, self.name, plan, critiques, evidence)


class SingleAgentResearcher:
    name = "single_agent"

    def __init__(
        self, llm: LLM, tracer: Tracer, tools: Toolbox, max_steps: int = 12, prompt_dir: Path | None = None
    ):
        self.llm, self.tracer, self.tb, self.max_steps = llm, tracer, tools, max_steps
        self.prompt = load_prompt("single_agent", prompt_dir)

    async def run(self, question: str) -> ResearchResult:
        with self.tracer.span("agent", "single_agent", role="single"):
            res = await run_tool_loop(
                TracedLLM(self.llm, self.tracer, "single"),
                self.prompt,
                question,
                self.tb,
                self.tb.tools,
                "single",
                max_steps=self.max_steps,
            )
            notes = ["hit step budget"] if res.hit_budget else []
            return ResearchResult(res.text, self.name, evidence=res.evidence, notes=notes)


def summarize_result(r: ResearchResult) -> dict[str, Any]:
    return {
        "answer": r.answer,
        "plan": r.plan,
        "critic": r.critic,
        "evidence_count": len(r.evidence),
        "notes": r.notes,
    }
