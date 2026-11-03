import json

from arp.agents.research import MultiAgentResearcher, SingleAgentResearcher
from arp.llm import LLMResponse, ToolCall
from arp.obs.replay import ReplayLLM, ReplayToolbox
from arp.obs.trace import Tracer, load_trace

TOOLS = [{"name": "pypi_package", "description": "PyPI metadata", "input_schema": {"type": "object"}}]
DATA = {"fastapi": {"requires_python": ">=3.10"}, "flask": {"requires_python": ">=3.9"}}


class FakeTools:
    tools = TOOLS

    def __init__(self, tracer=None):
        self.calls = []
        self.tracer = tracer

    def subset(self, names):
        return [t for t in TOOLS if t["name"] in names]

    async def call(self, agent, name, args):
        self.calls.append((agent, name, args))
        if args.get("name") not in DATA:
            text, err = "package or version not found on PyPI", True
        else:
            text, err = json.dumps(DATA[args["name"]]), False
        if self.tracer:  # mirrors MCPToolbox, which records each call as a span
            with self.tracer.span("tool", name, agent=agent, args=args) as sp:
                sp.set(result=text, is_error=err)
        return text, err


class FakeLLM:
    """Plays every role by looking at the system prompt, like a tiny deterministic model."""

    model = "fake"

    def __init__(self, critic_verdicts=("pass",)):
        self.critic_verdicts = list(critic_verdicts)

    async def chat(self, system, messages, tools=None, max_tokens=1024, json_mode=False):
        user = messages[0]["content"]
        if "planner" in system:
            return LLMResponse(
                json.dumps(
                    {
                        "subtasks": [
                            {"id": "t1", "question": "requires_python of fastapi"},
                            {"id": "t2", "question": "requires_python of flask"},
                        ]
                    }
                )
            )
        if "research worker" in system:
            if messages[-1]["role"] == "tool":
                return LLMResponse(f"FINDINGS:\n- {messages[-1]['content']}")
            pkg = "fastapi" if "fastapi" in user.split("(Context")[0] else "flask"
            return LLMResponse("", [ToolCall(f"c-{pkg}", "pypi_package", {"name": pkg})])
        if "final answer" in system:
            return LLMResponse("fastapi >=3.10, flask >=3.9\nANSWER: fastapi")
        if "verifier" in system:
            v = self.critic_verdicts.pop(0) if self.critic_verdicts else "pass"
            return LLMResponse(
                json.dumps(
                    {
                        "verdict": v,
                        "problems": ["x"] if v == "revise" else [],
                        "followups": ["check flask again"] if v == "revise" else [],
                    }
                )
            )
        # single agent
        if messages[-1]["role"] == "tool":
            return LLMResponse("ANSWER: fastapi")
        return LLMResponse(
            "",
            [
                ToolCall("a", "pypi_package", {"name": "fastapi"}),
                ToolCall("b", "pypi_package", {"name": "flask"}),
            ],
        )


async def test_multi_agent_plans_fans_out_and_verifies(tmp_path):
    tracer, tools = Tracer(tmp_path / "t.jsonl"), FakeTools()
    res = await MultiAgentResearcher(FakeLLM(), tracer, tools, tools).run("fastapi or flask?")
    assert res.answer.endswith("ANSWER: fastapi")
    assert len(res.plan) == 2 and {c[0] for c in tools.calls} == {"worker[0]", "worker[1]"}
    assert res.critic[0]["verdict"] == "pass"
    spans = load_trace(tmp_path / "t.jsonl")
    workers = [s for s in spans if s["kind"] == "agent" and s["name"].startswith("worker")]
    orch = next(s for s in spans if s["name"] == "multi_agent")
    assert all(w["parent_id"] == orch["span_id"] for w in workers)  # parallel spans nest correctly


async def test_critic_revision_runs_followup_workers(tmp_path):
    tools = FakeTools()
    res = await MultiAgentResearcher(FakeLLM(["revise", "pass"]), Tracer(), tools, tools).run(
        "fastapi or flask?"
    )
    assert [c["verdict"] for c in res.critic] == ["revise", "pass"]
    assert any(a.startswith("worker[r0") for a, _, _ in tools.calls)


async def test_planner_garbage_falls_back_to_single_task():
    class Bad(FakeLLM):
        async def chat(self, system, messages, tools=None, max_tokens=1024, json_mode=False):
            if "planner" in system:
                return LLMResponse("I think we should look it up")
            return await super().chat(system, messages, tools, max_tokens, json_mode)

    tools = FakeTools()
    res = await MultiAgentResearcher(Bad(), Tracer(), tools, tools).run("fastapi or flask?")
    assert len(res.plan) == 1


async def test_replay_reproduces_and_detects_divergence(tmp_path):
    path = tmp_path / "t.jsonl"
    tracer = Tracer(path)
    tools = FakeTools(tracer)
    with tracer.span("run", "q", question="fastapi or flask?"):
        original = await SingleAgentResearcher(FakeLLM(), tracer, tools).run("fastapi or flask?")
    spans = load_trace(path)
    replayed = await SingleAgentResearcher(ReplayLLM(spans), Tracer(), ReplayToolbox(spans, TOOLS)).run(
        "fastapi or flask?"
    )
    assert replayed.answer == original.answer
    import pytest

    from arp.obs.replay import Divergence

    with pytest.raises(Divergence):  # a different question means different requests
        await SingleAgentResearcher(ReplayLLM(spans), Tracer(), ReplayToolbox(spans, TOOLS)).run(
            "other question"
        )


async def test_tool_loop_dedupes_identical_calls():
    from arp.agents.loop import run_tool_loop

    class Repeat:
        model = "r"
        n = 0

        async def chat(self, system, messages, tools=None, max_tokens=1024, json_mode=False):
            self.n += 1
            if self.n <= 2:
                return LLMResponse("", [ToolCall(str(self.n), "pypi_package", {"name": "flask"})])
            return LLMResponse("done")

    tools = FakeTools()
    res = await run_tool_loop(Repeat(), "s", "u", tools, TOOLS, "w", max_steps=5)
    assert len(tools.calls) == 1 and res.repeated_calls == 1


async def test_step_budget_forces_answer():
    from arp.agents.loop import run_tool_loop

    class Forever:
        model = "f"

        async def chat(self, system, messages, tools=None, max_tokens=1024, json_mode=False):
            if not tools:
                return LLMResponse("best effort answer")
            return LLMResponse(
                "", [ToolCall(str(len(messages)), "pypi_package", {"name": f"p{len(messages)}"})]
            )

    res = await run_tool_loop(Forever(), "s", "u", FakeTools(), TOOLS, "w", max_steps=3)
    assert res.hit_budget and res.text == "best effort answer"


async def test_one_failed_worker_does_not_sink_the_run():
    class Overflow(FakeLLM):
        async def chat(self, system, messages, tools=None, max_tokens=1024, json_mode=False):
            if "research worker" in system and "flask" in messages[0]["content"].split("(Context")[0]:
                raise RuntimeError("request exceeds the available context size")
            return await super().chat(system, messages, tools, max_tokens, json_mode)

    tools = FakeTools()
    res = await MultiAgentResearcher(Overflow(), Tracer(), tools, tools).run("fastapi or flask?")
    assert res.answer.endswith("ANSWER: fastapi")
    assert {c[0] for c in tools.calls} == {"worker[0]"}
