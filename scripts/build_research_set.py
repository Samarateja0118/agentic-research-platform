"""Build evals/datasets/research.jsonl.

Every expected answer is computed from API responses, never typed by hand. The responses are
recorded into evals/cassettes/ at the same time, so agents later answer against the exact same
snapshot (``ARP_HTTP_MODE=replay``) and ground truth cannot drift.

    ARP_HTTP_MODE=record python scripts/build_research_set.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

from arp.mcp_server.backends import default_backends

OUT = Path("evals/datasets/research.jsonl")
HARD = Path("evals/datasets/research-hard.jsonl")
U = "Samarateja0118"
SNAP = "(PyPI/GitHub snapshot of 2026-11-02)"


def pyver(spec: str | None) -> tuple:
    m = re.search(r">=\s*(\d+)\.(\d+)", spec or "")
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def contains(*vals) -> list[dict]:
    return [{"type": "contains", "value": str(v)} for v in vals]


def first(winner: str, *others: str) -> dict:
    return {"type": "answer_first", "value": winner, "others": list(others)}


def yesno(flag: bool) -> dict:
    return {"type": "yesno", "value": "yes" if flag else "no"}


def number(n: int) -> dict:
    return {"type": "number", "value": int(n)}


async def main() -> None:
    os.environ.setdefault("ARP_HTTP_MODE", "record")
    pypi, gh = default_backends()
    P = {}

    async def pkg(name):
        if name not in P:
            P[name] = await pypi.package(name)
        return P[name]

    async def rel(name):
        return await pypi.releases(name, None, 100)

    qs: list[dict] = []

    def add(cat, q, checks, ref, hops):
        q = f"{q} {SNAP}"
        if any(c["type"] == "answer_first" for c in checks):
            q += " Begin the ANSWER line with the one you pick."
        if any(c["type"] == "yesno" for c in checks):
            q += " Begin the ANSWER line with yes or no."
        qs.append(
            {
                "id": f"q{len(qs) + 1:02d}",
                "category": cat,
                "hops": hops,
                "question": q,
                "checks": checks,
                "reference": ref,
            }
        )

    # --- single-entity lookups
    r = await pkg("rich")
    add(
        "lookup",
        "What is the latest version of the rich package on PyPI and what license does it declare?",
        contains(r["latest_version"], r["license"]),
        f"rich {r['latest_version']}, license {r['license']}",
        1,
    )
    r = await pkg("polars")
    add(
        "lookup",
        "What minimum Python version does the latest release of polars require?",
        contains(r["requires_python"].replace(">=", "").strip()),
        f"polars {r['latest_version']} requires {r['requires_python']}",
        1,
    )
    r = await pkg("ruff")
    add(
        "lookup",
        "On what date (YYYY-MM-DD) was the first release of ruff uploaded to PyPI?",
        contains(r["first_release_date"]),
        f"ruff first release {r['first_release_date']}",
        1,
    )
    rr = await rel("pydantic")
    n24 = rr["releases_per_year"].get("2024", 0)
    add(
        "lookup",
        "How many non-yanked releases (pre-releases included) did pydantic publish on PyPI in calendar year 2024?",
        [number(n24)],
        f"pydantic released {n24} versions in 2024",
        1,
    )
    r = await pkg("httpx")
    deps = [re.split(r"[<>=!~\[ ;]", d)[0] for d in r["dependencies"]]
    add(
        "lookup",
        "Which runtime dependencies (ignore optional extras) does the latest httpx release declare?",
        contains(*deps),
        f"httpx {r['latest_version']} depends on {', '.join(deps)}",
        1,
    )
    r = await pkg("typer")
    dep_click = any(d.lower().startswith("click") for d in r["dependencies"])
    add(
        "lookup",
        "Does the latest release of typer declare a runtime dependency on click? Answer yes or no and give the typer version.",
        contains(r["latest_version"]) + [yesno(dep_click)],
        f"typer {r['latest_version']}: {'depends' if dep_click else 'does not depend'} on click",
        1,
    )

    # --- comparisons across entities (fan-out)
    async def compare_python(a, b):
        pa, pb = await pkg(a), await pkg(b)
        va, vb = pyver(pa["requires_python"]), pyver(pb["requires_python"])
        assert va != vb, (a, b)
        win, lose = (a, b) if va > vb else (b, a)
        add(
            "compare",
            f"Which requires a newer minimum Python version in its latest release: {a} or {b}? Give both requires_python values.",
            [first(win, lose)] + contains(".".join(map(str, va)), ".".join(map(str, vb))),
            f"{a} {pa['requires_python']}, {b} {pb['requires_python']} -> {win} is newer",
            2,
        )

    await compare_python("fastapi", "flask")
    await compare_python("starlette", "httpx")

    a, b = await pkg("httpx"), await pkg("aiohttp")
    win, lose = (
        ("httpx", "aiohttp") if a["first_release_date"] < b["first_release_date"] else ("aiohttp", "httpx")
    )
    add(
        "compare",
        "Which was first released on PyPI earlier, httpx or aiohttp? Give both first-release dates (YYYY-MM-DD).",
        [first(win, lose)] + contains(a["first_release_date"], b["first_release_date"]),
        f"httpx {a['first_release_date']}, aiohttp {b['first_release_date']} -> {win} first",
        2,
    )

    counts = {n: (await rel(n))["releases_per_year"].get("2025", 0) for n in ("numpy", "pandas", "polars")}
    win = max(counts, key=counts.get)
    assert list(counts.values()).count(counts[win]) == 1
    add(
        "compare",
        "Among numpy, pandas and polars, which published the most non-yanked releases on PyPI in 2025? Give each count.",
        [first(win, *[k for k in counts if k != win])] + [number(v) for v in counts.values()],
        f"2025 releases: {counts} -> {win}",
        3,
    )

    ups = {n: (await pkg(n))["latest_upload_date"] for n in ("black", "mypy", "isort")}
    win = max(ups, key=ups.get)
    assert list(ups.values()).count(ups[win]) == 1
    add(
        "compare",
        "Of black, mypy and isort, which had the most recent release upload? Give each latest upload date (YYYY-MM-DD).",
        [first(win, *[k for k in ups if k != win])] + contains(*ups.values()),
        f"latest uploads {ups} -> {win}",
        3,
    )

    a, b = await pkg("sqlalchemy"), await pkg("django")
    win, lose = (
        ("sqlalchemy", "django") if a["release_count"] > b["release_count"] else ("django", "sqlalchemy")
    )
    diff = abs(a["release_count"] - b["release_count"])
    add(
        "compare",
        "Which has more non-yanked releases on PyPI, SQLAlchemy or Django, and by how many?",
        [first(win, lose), number(diff)] + contains(a["release_count"], b["release_count"]),
        f"SQLAlchemy {a['release_count']}, Django {b['release_count']}, difference {diff}",
        2,
    )

    rc = await rel("celery")
    c23, c24 = rc["releases_per_year"].get("2023", 0), rc["releases_per_year"].get("2024", 0)
    add(
        "compare",
        "How many non-yanked releases did celery publish in 2023 and in 2024, and in which of the two years were there more?",
        [number(c23), number(c24), first("2023" if c23 > c24 else "2024", "2024" if c23 > c24 else "2023")],
        f"celery 2023: {c23}, 2024: {c24}",
        1,
    )

    a, b = await pkg("fastapi"), await pkg("flask")
    na, nb = len(a["dependencies"]), len(b["dependencies"])
    assert na != nb
    win, lose = ("fastapi", "flask") if na > nb else ("flask", "fastapi")
    add(
        "compare",
        "Which declares more runtime dependencies (no extras) in its latest release, fastapi or flask? Give both counts.",
        [first(win, lose), number(na), number(nb)],
        f"fastapi {na} deps, flask {nb} deps",
        2,
    )

    a, b = await pkg("pytest"), await pkg("ruff")
    gap = int(b["first_release_date"][:4]) - int(a["first_release_date"][:4])
    add(
        "compare",
        "In which years were pytest and ruff first released on PyPI, and how many years apart are those years?",
        contains(a["first_release_date"][:4], b["first_release_date"][:4]) + [number(gap)],
        f"pytest {a['first_release_date'][:4]}, ruff {b['first_release_date'][:4]}, {gap} years apart",
        2,
    )

    a, b = await pkg("anthropic"), await pkg("mcp")
    win, lose = (
        ("anthropic", "mcp") if a["first_release_date"] < b["first_release_date"] else ("mcp", "anthropic")
    )
    add(
        "compare",
        "What are the latest versions of the anthropic and mcp packages, and which of the two appeared on PyPI first?",
        [first(win, lose)] + contains(a["latest_version"], b["latest_version"]),
        f"anthropic {a['latest_version']} (first {a['first_release_date']}), mcp {b['latest_version']} (first {b['first_release_date']})",
        2,
    )

    a, b = await pkg("starlette"), await pkg("click")
    add(
        "compare",
        "What licenses do the latest releases of starlette and click declare?",
        contains(a["license"], b["license"]),
        f"starlette {a['license']}, click {b['license']}",
        2,
    )

    # --- specific versions
    v, latest = await pypi.version("pydantic", "1.10.0"), await pkg("pydantic")
    add(
        "version",
        "What minimum Python version did pydantic 1.10.0 require, and what does the latest pydantic require?",
        contains(
            ".".join(map(str, pyver(v["requires_python"]))),
            ".".join(map(str, pyver(latest["requires_python"]))),
        ),
        f"pydantic 1.10.0 {v['requires_python']}; latest {latest['latest_version']} {latest['requires_python']}",
        2,
    )
    v, latest = await pypi.version("django", "4.2"), await pkg("django")
    add(
        "version",
        "What requires_python did Django 4.2 declare, and what is the latest Django version and its requires_python?",
        contains(
            ".".join(map(str, pyver(v["requires_python"]))),
            latest["latest_version"],
            ".".join(map(str, pyver(latest["requires_python"]))),
        ),
        f"Django 4.2 {v['requires_python']}; latest {latest['latest_version']} {latest['requires_python']}",
        2,
    )
    v, latest = await pypi.version("requests", "2.0.0"), await pkg("requests")
    add(
        "version",
        "On what date (YYYY-MM-DD) was requests 2.0.0 uploaded, and what is the latest requests version?",
        contains(v["upload_date"][:10], latest["latest_version"]),
        f"requests 2.0.0 uploaded {v['upload_date'][:10]}; latest {latest['latest_version']}",
        2,
    )
    v, latest = await pypi.version("numpy", "1.26.0"), await pkg("numpy")
    add(
        "version",
        "What is the latest numpy version, and what minimum Python version did numpy 1.26.0 require?",
        contains(latest["latest_version"], ".".join(map(str, pyver(v["requires_python"])))),
        f"numpy latest {latest['latest_version']}; 1.26.0 requires {v['requires_python']}",
        2,
    )

    # --- GitHub (the author's public repos)
    g = await gh.repo(U, "gpu-scheduler-platform")
    add(
        "github",
        f"What is the primary language and the license of the GitHub repo {U}/gpu-scheduler-platform?",
        contains(g["language"], g["license"]),
        f"{g['language']}, {g['license']}",
        1,
    )
    a, b = await gh.repo(U, "rag-eval-system"), await gh.repo(U, "mlb-mlops-pipeline")
    win, lose = (
        ("rag-eval-system", "mlb-mlops-pipeline")
        if a["created_at"] < b["created_at"]
        else ("mlb-mlops-pipeline", "rag-eval-system")
    )
    add(
        "github",
        f"Which GitHub repo was created first, {U}/rag-eval-system or {U}/mlb-mlops-pipeline? Give both creation dates (YYYY-MM-DD).",
        [first(win, lose)] + contains(a["created_at"][:10], b["created_at"][:10]),
        f"rag-eval-system {a['created_at']}, mlb-mlops-pipeline {b['created_at']}",
        2,
    )
    f = await gh.read_file(U, "rag-eval-system", ".github/workflows/ci.yml", 20000)
    pv = re.search(r'python-version:\s*"?([\d.]+)', f["content"]).group(1)
    add(
        "github",
        f"Which Python version does the CI workflow of {U}/rag-eval-system set up?",
        contains(pv),
        f"python-version {pv} in .github/workflows/ci.yml",
        2,
    )
    langs = await gh.languages(U, "movie-mcp-server")
    top = max(langs["percent"], key=langs["percent"].get)
    add(
        "github",
        f"How many languages does GitHub detect in {U}/movie-mcp-server, and which is the largest by bytes?",
        [number(len(langs["bytes"])), first(top, *[k for k in langs["bytes"] if k != top])],
        f"{len(langs['bytes'])} languages: {langs['percent']}",
        1,
    )
    runs = await gh.workflow_runs(U, "gpu-scheduler-platform", 10)
    ok = sum(1 for x in runs["runs"] if x["conclusion"] == "success")
    bad = sum(1 for x in runs["runs"] if x["conclusion"] == "failure")
    add(
        "github",
        f"Looking at the most recent GitHub Actions runs listed for {U}/gpu-scheduler-platform (up to 10), how many concluded with success and how many with failure?",
        [number(ok), number(bad)],
        f"{ok} success, {bad} failure: {[x['conclusion'] for x in runs['runs']]}",
        1,
    )
    d = await gh.list_dir(U, "mlb-mlops-pipeline", "")
    dirs = sorted(e["name"] for e in d["entries"] if e["type"] == "dir" and not e["name"].startswith("."))
    add(
        "github",
        f"List the top-level directories (ignore hidden ones) of the GitHub repo {U}/mlb-mlops-pipeline.",
        contains(*dirs),
        f"directories: {dirs}",
        1,
    )
    f = await gh.read_file(U, "gpu-scheduler-platform", "go.mod", 20000)
    gov = re.search(r"^go\s+([\d.]+)", f["content"], re.M).group(1)
    add(
        "github",
        f"Which Go version does {U}/gpu-scheduler-platform declare in its go.mod?",
        contains(gov),
        f"go {gov}",
        2,
    )

    # --- cross-source: read a repo file, then look up PyPI
    f = await gh.read_file(U, "movie-mcp-server", "requirements.txt", 20000)
    spec = re.search(r"^mcp([^\n]*)$", f["content"], re.M).group(1).strip()
    m = await pkg("mcp")
    major = int(m["latest_version"].split(".")[0])
    allowed = not ("<2" in spec and major >= 2)  # does the pinned range admit the latest release?
    add(
        "cross",
        f"What version range of the mcp package does {U}/movie-mcp-server pin in requirements.txt, and does that range allow the latest mcp release on PyPI?",
        contains(spec.split(",")[0].replace(">=", ""), m["latest_version"]) + [yesno(allowed)],
        f"pins mcp{spec}; latest mcp {m['latest_version']} -> {'allowed' if allowed else 'not allowed'}",
        3,
    )
    f = await gh.read_file(U, "mlb-mlops-pipeline", "pyproject.toml", 20000)
    mspec = re.search(r'"mlflow([^"]*)"', f["content"]).group(1)
    ml = await pkg("mlflow")
    add(
        "cross",
        f"What mlflow version range does {U}/mlb-mlops-pipeline require in pyproject.toml, and what is the latest mlflow version on PyPI?",
        contains(mspec.split(",")[0], ml["latest_version"]),
        f"requires mlflow{mspec}; latest {ml['latest_version']}",
        3,
    )
    f = await gh.read_file(U, "rag-eval-system", "pyproject.toml", 20000)
    rp = re.search(r'requires-python\s*=\s*"([^"]+)"', f["content"]).group(1)
    bm = await pkg("bm25s")
    add(
        "cross",
        f"What requires-python does {U}/rag-eval-system declare, and what minimum Python does the latest bm25s package (one of its dependencies) require?",
        contains(
            rp.split(",")[0].replace(">=", ""), (bm["requires_python"] or "none").replace(">=", "").strip()
        ),
        f"rag-eval-system {rp}; bm25s {bm['latest_version']} requires {bm['requires_python']}",
        3,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(q) + "\n" for q in qs))
    print(f"wrote {len(qs)} questions to {OUT}")
    for q in qs:
        print(q["id"], q["category"], "|", q["reference"])

    # --- hard set: 5-6 entities per question, so one agent's context fills with tool output
    main_qs = len(qs)
    rel24 = {n: (await rel(n))["releases_per_year"].get("2024", 0)
             for n in ("numpy", "pandas", "polars", "scipy", "matplotlib", "scikit-learn")}
    win = max(rel24, key=rel24.get)
    assert list(rel24.values()).count(rel24[win]) == 1
    add("hard", "For each of numpy, pandas, polars, scipy, matplotlib and scikit-learn, how many non-yanked releases were published on PyPI in 2024, and which published the most?",
        [first(win, *[k for k in rel24 if k != win])] + [number(v) for v in rel24.values()], f"2024 releases {rel24}", 6)
    rp = {n: (await pkg(n))["requires_python"] for n in ("django", "flask", "fastapi", "httpx", "aiohttp", "celery")}
    win = max(rp, key=lambda n: pyver(rp[n]))
    assert [pyver(v) for v in rp.values()].count(pyver(rp[win])) == 1
    add("hard", "Give the minimum Python version required by the latest release of each of django, flask, fastapi, httpx, aiohttp and celery. Which one requires the newest Python?",
        [first(win, *[k for k in rp if k != win])] + contains(*sorted({".".join(map(str, pyver(v))) for v in rp.values()})),
        f"requires_python {rp}", 6)
    fy = {n: (await pkg(n))["first_release_date"][:4] for n in ("pytest", "ruff", "black", "mypy", "isort", "flake8")}
    win = min(fy, key=fy.get)
    assert list(fy.values()).count(fy[win]) == 1
    add("hard", "In which year was each of pytest, ruff, black, mypy, isort and flake8 first released on PyPI? Which is the oldest?",
        [first(win, *[k for k in fy if k != win])] + contains(*sorted(set(fy.values()))), f"first release years {fy}", 6)
    langs = {r: (await gh.repo(U, r))["language"] for r in ("gpu-scheduler-platform", "rag-eval-system", "mlb-mlops-pipeline", "movie-mcp-server")}
    npy = sum(1 for v in langs.values() if v == "Python")
    add("hard", f"What is the primary language of each of these GitHub repos: {U}/gpu-scheduler-platform, {U}/rag-eval-system, {U}/mlb-mlops-pipeline and {U}/movie-mcp-server? How many of them are primarily Python?",
        contains(*sorted(set(langs.values()))) + [number(npy)], f"languages {langs}, {npy} Python", 4)
    lv = {n: (await pkg(n))["latest_version"] for n in ("anthropic", "mcp", "pydantic", "httpx", "fastapi", "uvicorn")}
    add("hard", "What is the latest version on PyPI of each of anthropic, mcp, pydantic, httpx, fastapi and uvicorn?",
        contains(*lv.values()), f"latest versions {lv}", 6)
    tot = {n: (await pkg(n))["release_count"] for n in ("requests", "httpx", "aiohttp", "urllib3", "certifi")}
    win = max(tot, key=tot.get)
    assert list(tot.values()).count(tot[win]) == 1
    add("hard", "How many non-yanked releases has each of requests, httpx, aiohttp, urllib3 and certifi published on PyPI in total, and which has the most?",
        [first(win, *[k for k in tot if k != win])] + [number(v) for v in tot.values()], f"total releases {tot}", 5)
    hard = qs[main_qs:]
    for i, q in enumerate(hard):
        q["id"] = f"h{i + 1:02d}"
    HARD.write_text("".join(json.dumps(q) + "\n" for q in hard))
    print(f"wrote {len(hard)} questions to {HARD}")
    for q in hard:
        print(q["id"], "|", q["reference"])


if __name__ == "__main__":
    asyncio.run(main())
