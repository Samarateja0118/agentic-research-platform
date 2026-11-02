"""Markdown comparison of eval result files."""

from __future__ import annotations

from .stats import paired_bootstrap


def compare_markdown(docs: list[dict]) -> str:
    cols = [d["meta"].get("label") or d["meta"]["system"] for d in docs]
    metrics = [
        ("questions", "n", "{}"),
        ("pass rate (all checks)", "pass_rate", "{:.0%}"),
        ("checks passed", "check_fraction", "{:.0%}"),
        ("judge mean (1-5)", "judge_mean", "{}"),
        ("judge >= 4", "judge_ge4", "{:.0%}"),
        ("crashed runs", "errors", "{}"),
        ("LLM calls / question", "mean_llm_calls", "{}"),
        ("tool calls / question", "mean_tool_calls", "{}"),
        ("tool error rate", "tool_error_rate", "{:.1%}"),
        ("input tokens / question", "mean_input_tokens", "{:,}"),
        ("output tokens / question", "mean_output_tokens", "{:,}"),
        ("wall time / question (s)", "mean_wall_s", "{}"),
        ("$ / question at Haiku 5.5 prices", "mean_cost_at_claude-haiku-5-5", "${:.4f}"),
        ("$ / question at Sonnet 5.5 prices", "mean_cost_at_claude-sonnet-5-5", "${:.4f}"),
    ]
    out = ["| metric | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for title, key, fmt in metrics:
        vals = [d["summary"].get(key) for d in docs]
        out.append(f"| {title} | " + " | ".join("-" if v is None else fmt.format(v) for v in vals) + " |")
    cats = sorted({c for d in docs for c in d["summary"]["pass_by_category"]})
    for c in cats:
        out.append(
            f"| pass: {c} | " + " | ".join(d["summary"]["pass_by_category"].get(c, "-") for d in docs) + " |"
        )
    if len(docs) >= 2:
        a, b = ({r["id"]: r for r in d["rows"]} for d in docs[:2])
        ids = sorted(set(a) & set(b))
        if ids:
            d, lo, hi = paired_bootstrap(
                [float(a[i]["pass"]) for i in ids], [float(b[i]["pass"]) for i in ids]
            )
            out.append("")
            out.append(
                f"Paired difference in pass rate ({cols[1]} minus {cols[0]}): {d:+.3f}, "
                f"95% bootstrap CI [{lo:+.3f}, {hi:+.3f}] over {len(ids)} questions."
            )
            only_a = [i for i in ids if a[i]["pass"] and not b[i]["pass"]]
            only_b = [i for i in ids if b[i]["pass"] and not a[i]["pass"]]
            out.append(
                f"Only {cols[0]} passes: {', '.join(only_a) or 'none'}. Only {cols[1]} passes: {', '.join(only_b) or 'none'}."
            )
    return "\n".join(out)
