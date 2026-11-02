"""PyPI and GitHub REST backends. Each method returns a compact dict meant for an LLM to read."""

from __future__ import annotations

import base64
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .http import Cassette, HttpBackend, UpstreamError, backend_mode

PRE_RE = re.compile(r"(a|b|rc|dev)\d*", re.I)


def is_prerelease(version: str) -> bool:
    return bool(PRE_RE.search(version))


def release_rows(releases: dict[str, list[dict]]) -> list[dict]:
    """A release counts if it has at least one uploaded file. Date = earliest upload."""
    rows = []
    for version, files in releases.items():
        if not files:
            continue
        date = min(f["upload_time_iso_8601"] for f in files)[:10]
        rows.append(
            {
                "version": version,
                "date": date,
                "yanked": all(f.get("yanked") for f in files),
                "prerelease": is_prerelease(version),
            }
        )
    rows.sort(key=lambda r: (r["date"], r["version"]), reverse=True)
    return rows


def _plain_requirements(requires_dist: list[str] | None) -> list[str]:
    out = []
    for req in requires_dist or []:
        if "extra ==" in req:
            continue
        out.append(req.split(";")[0].strip())
    return out


def shrink_pypi(url: str, body: Any) -> Any:
    """Keep only the fields the tools read. Full PyPI JSON for Django is several MB."""
    if not isinstance(body, dict) or "info" not in body:
        return body
    keep = (
        "name",
        "version",
        "summary",
        "license",
        "license_expression",
        "requires_python",
        "requires_dist",
        "project_urls",
        "home_page",
    )

    def squash(files: list[dict]) -> list[dict]:
        if not files:
            return []
        return [
            {
                "upload_time_iso_8601": min(f["upload_time_iso_8601"] for f in files),
                "yanked": all(f.get("yanked") for f in files),
            }
        ]

    out = {"info": {k: body["info"].get(k) for k in keep}, "urls": squash(body.get("urls") or [])}
    if "releases" in body and url.count("/") == 5:  # /pypi/<name>/json only
        out["releases"] = {v: squash(files) for v, files in body["releases"].items()}
    return out


class PyPIBackend:
    BASE = "https://pypi.org/pypi"

    def __init__(self, http: HttpBackend):
        self.http = http

    async def _get(self, url: str) -> dict:
        res = await self.http.get_json(url)
        if res.status == 404:
            raise LookupError("package or version not found on PyPI")
        if res.status >= 400:
            raise UpstreamError(f"PyPI HTTP {res.status}", res.status, res.attempts)
        return res.body

    async def package(self, name: str) -> dict[str, Any]:
        d = await self._get(f"{self.BASE}/{quote(name.strip().lower())}/json")
        info = d["info"]
        rows = [r for r in release_rows(d["releases"]) if not r["yanked"]]
        urls = info.get("project_urls") or {}
        return {
            "name": info["name"],
            "latest_version": info["version"],
            "summary": info.get("summary"),
            "license": info.get("license_expression") or (info.get("license") or "")[:80] or None,
            "requires_python": info.get("requires_python"),
            "first_release_date": rows[-1]["date"] if rows else None,
            "latest_upload_date": rows[0]["date"] if rows else None,
            "release_count": len(rows),
            "dependencies": _plain_requirements(info.get("requires_dist"))[:30],
            "source_url": urls.get("Source") or urls.get("Homepage") or info.get("home_page"),
        }

    async def releases(self, name: str, year: int | None = None, limit: int = 40) -> dict[str, Any]:
        d = await self._get(f"{self.BASE}/{quote(name.strip().lower())}/json")
        rows = [r for r in release_rows(d["releases"]) if not r["yanked"]]
        by_year: dict[str, int] = {}
        for r in rows:
            by_year[r["date"][:4]] = by_year.get(r["date"][:4], 0) + 1
        if year:
            rows = [r for r in rows if r["date"].startswith(str(year))]
        limit = max(1, min(int(limit), 100))
        return {
            "name": d["info"]["name"],
            "total_releases": sum(by_year.values()),
            "releases_per_year": dict(sorted(by_year.items())),
            "filter_year": year,
            "matching": len(rows),
            "releases_newest_first": [
                {"version": r["version"], "date": r["date"], "prerelease": r["prerelease"]}
                for r in rows[:limit]
            ],
            "truncated": len(rows) > limit,
        }

    async def version(self, name: str, version: str) -> dict[str, Any]:
        d = await self._get(f"{self.BASE}/{quote(name.strip().lower())}/{quote(version.strip())}/json")
        info = d["info"]
        files = d.get("urls") or []  # already squashed to one row by shrink_pypi
        return {
            "name": info["name"],
            "version": info["version"],
            "requires_python": info.get("requires_python"),
            "dependencies": _plain_requirements(info.get("requires_dist"))[:30],
            "upload_date": min((f["upload_time_iso_8601"] for f in files), default=None),
            "yanked": bool(files) and all(f.get("yanked") for f in files),
        }


def shrink_github(url: str, body: Any) -> Any:
    if "/actions/runs" in url and isinstance(body, dict):
        keep = ("name", "event", "status", "conclusion", "head_branch", "head_sha", "created_at")
        return {
            "total_count": body.get("total_count"),
            "workflow_runs": [{k: r.get(k) for k in keep} for r in body.get("workflow_runs", [])],
        }
    if "/commits" in url and isinstance(body, list):
        return [
            {
                "sha": c["sha"],
                "commit": {
                    "message": c["commit"]["message"],
                    "committer": {"date": c["commit"]["committer"]["date"]},
                },
            }
            for c in body
        ]
    if isinstance(body, dict) and "full_name" in body and "owner" in body:
        keep = (
            "full_name",
            "description",
            "language",
            "license",
            "created_at",
            "pushed_at",
            "default_branch",
            "stargazers_count",
            "open_issues_count",
            "topics",
            "size",
        )
        return {k: body.get(k) for k in keep}
    if isinstance(body, list) and body and "download_url" in body[0]:
        return [{"name": e["name"], "type": e["type"], "size": e.get("size")} for e in body]
    if isinstance(body, dict) and body.get("type") == "file":
        return {k: body.get(k) for k in ("type", "size", "content", "sha", "path")}
    return body


class GitHubBackend:
    BASE = "https://api.github.com"

    def __init__(self, http: HttpBackend):
        self.http = http

    async def _get(self, path: str) -> Any:
        res = await self.http.get_json(f"{self.BASE}{path}")
        if res.status == 404:
            raise LookupError(f"not found on GitHub: {path}")
        if res.status >= 400:
            raise UpstreamError(f"GitHub HTTP {res.status}: {str(res.body)[:200]}", res.status, res.attempts)
        return res.body

    async def repo(self, owner: str, repo: str) -> dict[str, Any]:
        d = await self._get(f"/repos/{owner}/{repo}")
        return {
            "full_name": d["full_name"],
            "description": d.get("description"),
            "language": d.get("language"),
            "license": (d.get("license") or {}).get("spdx_id"),
            "created_at": d.get("created_at"),
            "pushed_at": d.get("pushed_at"),
            "default_branch": d.get("default_branch"),
            "stars": d.get("stargazers_count"),
            "open_issues": d.get("open_issues_count"),
            "topics": d.get("topics"),
            "size_kb": d.get("size"),
        }

    async def languages(self, owner: str, repo: str) -> dict[str, Any]:
        d = await self._get(f"/repos/{owner}/{repo}/languages")
        total = sum(d.values()) or 1
        return {"bytes": d, "percent": {k: round(100 * v / total, 1) for k, v in d.items()}}

    async def list_dir(self, owner: str, repo: str, path: str = "") -> dict[str, Any]:
        sub = quote(path.strip("/"))
        d = await self._get(f"/repos/{owner}/{repo}/contents" + (f"/{sub}" if sub else ""))
        if isinstance(d, dict):
            return {"path": path, "note": "this path is a file, use github_read_file"}
        return {
            "path": path or "/",
            "entries": [{"name": e["name"], "type": e["type"], "size": e.get("size")} for e in d][:200],
        }

    async def read_file(self, owner: str, repo: str, path: str, max_chars: int = 6000) -> dict[str, Any]:
        d = await self._get(f"/repos/{owner}/{repo}/contents/{quote(path.strip('/'))}")
        if isinstance(d, list):
            return {"path": path, "note": "this path is a directory, use github_list_dir"}
        text = base64.b64decode(d.get("content", "")).decode("utf-8", errors="replace")
        max_chars = max(200, min(int(max_chars), 20000))
        return {
            "path": path,
            "size": d.get("size"),
            "content": text[:max_chars],
            "truncated": len(text) > max_chars,
        }

    async def workflow_runs(self, owner: str, repo: str, limit: int = 10) -> dict[str, Any]:
        d = await self._get(f"/repos/{owner}/{repo}/actions/runs?per_page=30")
        runs = d.get("workflow_runs", [])
        limit = max(1, min(int(limit), 30))
        return {
            "total_count": d.get("total_count"),
            "runs": [
                {
                    "workflow": r.get("name"),
                    "event": r.get("event"),
                    "status": r.get("status"),
                    "conclusion": r.get("conclusion"),
                    "branch": r.get("head_branch"),
                    "sha": (r.get("head_sha") or "")[:7],
                    "created_at": r.get("created_at"),
                }
                for r in runs[:limit]
            ],
        }

    async def commits(self, owner: str, repo: str, limit: int = 10) -> dict[str, Any]:
        d = await self._get(f"/repos/{owner}/{repo}/commits?per_page=30")
        limit = max(1, min(int(limit), 30))
        return {
            "commits": [
                {
                    "sha": c["sha"][:7],
                    "date": c["commit"]["committer"]["date"],
                    "message": c["commit"]["message"].splitlines()[0][:120],
                }
                for c in d[:limit]
            ]
        }

    async def create_pull_request(
        self, owner: str, repo: str, base: str, branch: str, title: str, body: str, files: dict[str, str]
    ) -> dict[str, Any]:
        """Create branch from base, commit each file through the contents API, open a PR."""
        ref = await self.http.get_json(f"{self.BASE}/repos/{owner}/{repo}/git/ref/heads/{base}")
        if ref.status >= 400:
            raise UpstreamError(f"base branch lookup failed: HTTP {ref.status}", ref.status)
        sha = ref.body["object"]["sha"]
        r = await self.http.send_json(
            "POST", f"{self.BASE}/repos/{owner}/{repo}/git/refs", {"ref": f"refs/heads/{branch}", "sha": sha}
        )
        if r.status >= 400:
            raise UpstreamError(f"create branch failed: HTTP {r.status}: {str(r.body)[:200]}", r.status)
        for path, content in files.items():
            existing = await self.http.get_json(
                f"{self.BASE}/repos/{owner}/{repo}/contents/{quote(path)}?ref={branch}"
            )
            payload = {
                "message": f"{title}\n\n{path}",
                "branch": branch,
                "content": base64.b64encode(content.encode()).decode(),
            }
            if existing.status == 200 and isinstance(existing.body, dict):
                payload["sha"] = existing.body["sha"]
            w = await self.http.send_json(
                "PUT", f"{self.BASE}/repos/{owner}/{repo}/contents/{quote(path)}", payload
            )
            if w.status >= 400:
                raise UpstreamError(f"commit {path} failed: HTTP {w.status}", w.status)
        pr = await self.http.send_json(
            "POST",
            f"{self.BASE}/repos/{owner}/{repo}/pulls",
            {"title": title, "head": branch, "base": base, "body": body, "draft": True},
        )
        if pr.status >= 400:
            raise UpstreamError(f"open PR failed: HTTP {pr.status}: {str(pr.body)[:200]}", pr.status)
        return {"number": pr.body["number"], "url": pr.body["html_url"], "draft": True}


def default_backends(cassette_dir: str | Path | None = None, mode: str | None = None):
    mode = mode or backend_mode()
    cdir = Path(cassette_dir or os.environ.get("ARP_CASSETTE_DIR", "evals/cassettes"))
    gh_headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "agentic-research-platform",
    }
    if os.environ.get("GITHUB_TOKEN"):
        gh_headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    pypi = HttpBackend(
        "pypi",
        mode,
        Cassette(cdir / "pypi.json"),
        {"User-Agent": "agentic-research-platform"},
        shrink=shrink_pypi,
    )
    gh = HttpBackend("github", mode, Cassette(cdir / "github.json"), gh_headers, shrink=shrink_github)
    return PyPIBackend(pypi), GitHubBackend(gh)
