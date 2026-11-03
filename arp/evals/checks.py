"""Deterministic answer checks. A question passes only if every check passes."""

from __future__ import annotations

import re
from typing import Any

NUMBER_WORDS = {
    w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())
}


def normalize(text: str) -> str:
    text = text.lower().replace("‑", "-").replace("–", "-").replace("**", "").replace("`", "")
    return re.sub(r"\s+", " ", text)


def answer_line(text: str) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip().lower().lstrip("*# ").startswith("answer")]
    return normalize(lines[-1]) if lines else normalize(text.strip().splitlines()[-1] if text.strip() else "")


def _has_token(hay: str, needle: str) -> bool:
    """Match needle not glued to other digits/letters, so '3.1' does not match '3.12'."""
    pat = r"(?<![\w.])" + re.escape(needle) + r"(?![\w]|\.\d)"
    return re.search(pat, hay) is not None


def run_check(check: dict[str, Any], answer: str) -> tuple[bool, str]:
    full, line = normalize(answer), answer_line(answer)
    kind, value = check["type"], str(check["value"]).lower()
    if kind == "contains":
        ok = _has_token(full, value) or _has_token(full.replace(",", ""), value)
        return ok, f"contains '{value}'"
    if kind == "number":
        nums = {n.replace(",", "") for n in re.findall(r"(?<![\w.])\d[\d,]*(?![\w.]\d)", full)}
        nums |= {str(i) for w, i in NUMBER_WORDS.items() if re.search(rf"\b{w}\b", full)}
        ok = value in nums
        return ok, f"number {value}"
    if kind == "yesno":
        other = "no" if value == "yes" else "yes"
        has, has_other = bool(re.search(rf"\b{value}\b", line)), bool(re.search(rf"\b{other}\b", line))
        return has and not has_other, f"ANSWER line says '{value}'"
    if kind == "answer_first":
        pos = line.find(value)
        others = [line.find(o.lower()) for o in check.get("others", [])]
        ok = pos != -1 and all(o == -1 or pos < o for o in others)
        return ok, f"ANSWER line names '{value}' first"
    raise ValueError(f"unknown check type {kind}")


def score(checks: list[dict], answer: str) -> dict[str, Any]:
    results = [(c, *run_check(c, answer)) for c in checks]
    passed = sum(1 for _, ok, _ in results if ok)
    return {
        "pass": passed == len(results),
        "check_fraction": round(passed / len(results), 3) if results else 1.0,
        "failed_checks": [desc for _, ok, desc in results if not ok],
    }
