import json

from arp.agents.research import MULTI_PROMPTS, prompt_version
from arp.evals.gate import gate


def doc(passes, pv, judge=5):
    return {
        "meta": {
            "system": "multi_agent",
            "label": "x",
            "prompt_version": pv,
            "dataset_digest": "d",
            "snapshot_digest": {"pypi": "s"},
        },
        "rows": [
            {"id": f"q{i}", "pass": p, "judge": {"score": judge}, "input_tokens": 100, "output_tokens": 10}
            for i, p in enumerate(passes)
        ],
    }


def write(tmp_path, name, d):
    p = tmp_path / name
    p.write_text(json.dumps(d))
    return str(p)


def test_gate_passes_on_equal_quality(tmp_path):
    pv = prompt_version(MULTI_PROMPTS)
    ok, report = gate(
        write(tmp_path, "b.json", doc([1] * 10, pv)), write(tmp_path, "c.json", doc([1] * 10, pv))
    )
    assert ok, report


def test_gate_fails_on_drop(tmp_path):
    pv = prompt_version(MULTI_PROMPTS)
    ok, report = gate(
        write(tmp_path, "b.json", doc([1] * 10, pv)), write(tmp_path, "c.json", doc([1] * 8 + [0] * 2, pv))
    )
    assert not ok and "pass rate dropped" in report


def test_gate_fails_on_stale_prompts(tmp_path):
    ok, report = gate(
        write(tmp_path, "b.json", doc([1] * 10, "old")), write(tmp_path, "c.json", doc([1] * 10, "old"))
    )
    assert not ok and "stale results" in report
