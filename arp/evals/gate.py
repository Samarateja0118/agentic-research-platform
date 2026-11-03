"""Regression gate for prompt and agent changes, run in CI.

The gate fails when:
1. the candidate results were produced with prompts that differ from the prompts in the repo
   (a prompt was edited but the eval was not re-run), or on a different dataset or snapshot;
2. pass rate drops by more than ``max_drop`` versus the baseline, or the judge mean drops by
   more than 0.3 points.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..agents.research import MULTI_PROMPTS, SINGLE_PROMPTS, prompt_version
from .checks import score
from .runner import load_dataset
from .stats import paired_bootstrap


def _load(p):
    return json.loads(Path(p).read_text())


def gate(
    baseline: str, candidate: str, max_drop: float = 0.05, prompts_dir: str = "prompts"
) -> tuple[bool, str]:
    b, c = _load(baseline), _load(candidate)
    lines, ok = (
        [
            f"baseline:  {baseline} ({b['meta'].get('label')})",
            f"candidate: {candidate} ({c['meta'].get('label')})",
        ],
        True,
    )
    names = MULTI_PROMPTS if c["meta"]["system"] == "multi_agent" else SINGLE_PROMPTS
    current = prompt_version(names, Path(prompts_dir))
    if c["meta"]["prompt_version"] != current:
        ok = False
        lines.append(
            f"FAIL stale results: prompts in {prompts_dir} hash to {current}, candidate was run with "
            f"{c['meta']['prompt_version']}. Re-run `arp eval` and commit the new results."
        )
    for key in ("dataset_digest", "snapshot_digest", "system"):
        if b["meta"].get(key) != c["meta"].get(key):
            ok = False
            lines.append(f"FAIL not comparable: {key} differs ({b['meta'].get(key)} vs {c['meta'].get(key)})")
    bm, cm = {r["id"]: r for r in b["rows"]}, {r["id"]: r for r in c["rows"]}
    ids = sorted(set(bm) & set(cm))
    if len(ids) < len(bm):
        ok = False
        lines.append(f"FAIL candidate covers {len(ids)}/{len(bm)} baseline questions")
    if ids:
        pa, pc = [float(bm[i]["pass"]) for i in ids], [float(cm[i]["pass"]) for i in ids]
        d, lo, hi = paired_bootstrap(pa, pc)
        lines.append(
            f"pass rate {sum(pa) / len(ids):.3f} -> {sum(pc) / len(ids):.3f} "
            f"(delta {d:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}], n={len(ids)})"
        )
        if d < -max_drop:
            ok = False
            lines.append(f"FAIL pass rate dropped by more than {max_drop:.2f}")
        jb = [bm[i].get("judge", {}).get("score") for i in ids]
        jc = [cm[i].get("judge", {}).get("score") for i in ids]
        pairs = [(x, y) for x, y in zip(jb, jc, strict=True) if x and y]
        if pairs:
            dj = sum(y - x for x, y in pairs) / len(pairs)
            lines.append(
                f"judge mean {sum(x for x, _ in pairs) / len(pairs):.2f} -> "
                f"{sum(y for _, y in pairs) / len(pairs):.2f} (delta {dj:+.2f})"
            )
            if dj < -0.3:
                ok = False
                lines.append("FAIL judge mean dropped by more than 0.3")
        broke = [i for i in ids if bm[i]["pass"] and not cm[i]["pass"]]
        fixed = [i for i in ids if not bm[i]["pass"] and cm[i]["pass"]]
        lines.append(
            f"newly failing: {', '.join(broke) or 'none'}; newly passing: {', '.join(fixed) or 'none'}"
        )
        tb = sum(bm[i]["input_tokens"] + bm[i]["output_tokens"] for i in ids)
        tc = sum(cm[i]["input_tokens"] + cm[i]["output_tokens"] for i in ids)
        lines.append(
            f"tokens per question {tb / len(ids):.0f} -> {tc / len(ids):.0f} ({(tc - tb) / max(tb, 1):+.0%})"
        )
    lines.append("GATE: PASS" if ok else "GATE: FAIL")
    return ok, "\n".join(lines)


def rescore(results: str, dataset: str, write: bool = False) -> tuple[bool, str]:
    """Re-apply the current deterministic checks to the recorded answers. A mismatch means the
    scorer or dataset changed without re-running the eval."""
    doc = _load(results)
    qs = {q["id"]: q for q in load_dataset(dataset)}
    bad = []
    for r in doc["rows"]:
        s = score(qs[r["id"]]["checks"], r["answer"])
        if s["pass"] != r["pass"]:
            bad.append(
                f"{r['id']}: recorded pass={r['pass']} but current checks give {s['pass']} {s['failed_checks']}"
            )
        if write:
            r.update(s)
    if write:
        from .runner import summarize

        doc["summary"] = summarize(doc["rows"])
        Path(results).write_text(json.dumps(doc, indent=1, default=str))
        return True, "\n".join(bad + [f"rewrote {results} with the current checks"])
    return not bad, "\n".join(bad) or f"rescore OK: {len(doc['rows'])} answers score as recorded"
