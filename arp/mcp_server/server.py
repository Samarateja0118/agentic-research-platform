"""MCP server exposing PyPI and GitHub tools behind a permission policy, approvals and an audit log.

Run: ``ARP_PRINCIPAL=researcher python -m arp.mcp_server.server`` (stdio transport).
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import yaml
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .backends import GitHubBackend, PyPIBackend, default_backends
from .http import UpstreamError
from .policy import ApprovalStore, AuditLog, Policy

READ_ONLY = {"readOnlyHint": True, "openWorldHint": True}
DESC = yaml.safe_load((Path(__file__).resolve().parents[2] / "prompts" / "tools.yaml").read_text())


class Guard:
    """Wraps every tool: policy check -> approval check -> call -> audit line."""

    def __init__(
        self, principal: str, policy: Policy, audit: AuditLog, approvals: ApprovalStore, session: str
    ):
        self.principal = principal
        self.policy = policy
        self.audit = audit
        self.approvals = approvals
        self.session = session
        self.calls = 0

    async def run(
        self,
        tool: str,
        args: dict[str, Any],
        fn: Callable[[], Awaitable[dict]],
        approval_payload: dict | None = None,
    ) -> dict:
        start = time.perf_counter()
        decision = self.policy.check(self.principal, tool, args, self.calls)
        base = {"session": self.session, "principal": self.principal, "tool": tool, "args": args}
        if not decision.allowed:
            self.audit.write(**base, decision="deny", reason=decision.reason, rule=decision.rule)
            raise ToolError(f"permission denied: {decision.reason}")
        if self.policy.needs_approval(self.principal, tool):
            ok, why = self.approvals.is_approved(approval_payload or args)
            if not ok:
                req = self.approvals.request(approval_payload or args, f"{self.principal} -> {tool}")
                self.audit.write(**base, decision="pending_approval", reason=why, approval_request=req)
                raise ToolError(f"human approval required: run `arp approve {req}` and retry")
        self.calls += 1
        try:
            result = await fn()
        except LookupError as exc:
            self.audit.write(
                **base,
                decision="allow",
                status="not_found",
                error=str(exc),
                latency_ms=round((time.perf_counter() - start) * 1e3, 1),
            )
            raise ToolError(str(exc)) from exc
        except (UpstreamError, ValueError, KeyError, TypeError) as exc:
            self.audit.write(
                **base,
                decision="allow",
                status="error",
                error=f"{type(exc).__name__}: {exc}",
                attempts=getattr(exc, "attempts", None),
                latency_ms=round((time.perf_counter() - start) * 1e3, 1),
            )
            raise ToolError(f"upstream error: {exc}") from exc
        self.audit.write(
            **base, decision="allow", status="ok", latency_ms=round((time.perf_counter() - start) * 1e3, 1)
        )
        return result


def build_server(
    principal: str,
    policy: Policy,
    audit: AuditLog,
    approvals: ApprovalStore,
    pypi: PyPIBackend,
    gh: GitHubBackend,
) -> MCPServer:
    server = MCPServer(
        "arp-research-tools",
        instructions="PyPI and GitHub research tools. Read-only "
        "except github_create_pull_request, which needs human approval.",
    )
    g = Guard(principal, policy, audit, approvals, session=uuid.uuid4().hex[:8])

    @server.tool(annotations=READ_ONLY, description=DESC["pypi_package"])
    async def pypi_package(name: str) -> dict:
        """PyPI metadata for a package: latest version, license, requires_python, first and latest
        upload dates, number of non-yanked releases, and runtime dependencies (extras excluded)."""
        return await g.run("pypi_package", {"name": name}, lambda: pypi.package(name))

    @server.tool(annotations=READ_ONLY, description=DESC["pypi_releases"])
    async def pypi_releases(name: str, year: int | None = None, limit: int = 40) -> dict:
        """Release history of a PyPI package, newest first. Includes total_releases and
        releases_per_year (non-yanked releases, pre-releases included, dated by first upload).
        Pass year to list only that calendar year."""
        return await g.run(
            "pypi_releases",
            {"name": name, "year": year, "limit": limit},
            lambda: pypi.releases(name, year, limit),
        )

    @server.tool(annotations=READ_ONLY, description=DESC["pypi_version"])
    async def pypi_version(name: str, version: str) -> dict:
        """Metadata for one specific version of a PyPI package: requires_python, dependencies, upload date."""
        return await g.run(
            "pypi_version", {"name": name, "version": version}, lambda: pypi.version(name, version)
        )

    @server.tool(annotations=READ_ONLY, description=DESC["github_repo"])
    async def github_repo(owner: str, repo: str) -> dict:
        """GitHub repository metadata: description, primary language, license, created/pushed dates, stars."""
        return await g.run("github_repo", {"owner": owner, "repo": repo}, lambda: gh.repo(owner, repo))

    @server.tool(annotations=READ_ONLY, description=DESC["github_languages"])
    async def github_languages(owner: str, repo: str) -> dict:
        """Bytes and percent of code per language in a GitHub repository."""
        return await g.run(
            "github_languages", {"owner": owner, "repo": repo}, lambda: gh.languages(owner, repo)
        )

    @server.tool(annotations=READ_ONLY, description=DESC["github_list_dir"])
    async def github_list_dir(owner: str, repo: str, path: str = "") -> dict:
        """List files and folders at a path in a GitHub repository (default branch). Empty path = root."""
        return await g.run(
            "github_list_dir",
            {"owner": owner, "repo": repo, "path": path},
            lambda: gh.list_dir(owner, repo, path),
        )

    @server.tool(annotations=READ_ONLY, description=DESC["github_read_file"])
    async def github_read_file(owner: str, repo: str, path: str, max_chars: int = 6000) -> dict:
        """Read a text file from a GitHub repository (default branch), truncated to max_chars."""
        return await g.run(
            "github_read_file",
            {"owner": owner, "repo": repo, "path": path},
            lambda: gh.read_file(owner, repo, path, max_chars),
        )

    @server.tool(annotations=READ_ONLY, description=DESC["github_workflow_runs"])
    async def github_workflow_runs(owner: str, repo: str, limit: int = 10) -> dict:
        """Recent GitHub Actions runs (newest first) with workflow name, event, conclusion and branch."""
        return await g.run(
            "github_workflow_runs",
            {"owner": owner, "repo": repo, "limit": limit},
            lambda: gh.workflow_runs(owner, repo, limit),
        )

    @server.tool(annotations=READ_ONLY, description=DESC["github_commits"])
    async def github_commits(owner: str, repo: str, limit: int = 10) -> dict:
        """Recent commits on the default branch (newest first): short sha, date, first line of message."""
        return await g.run(
            "github_commits",
            {"owner": owner, "repo": repo, "limit": limit},
            lambda: gh.commits(owner, repo, limit),
        )

    @server.tool(
        annotations={"destructiveHint": False, "openWorldHint": True},
        description=DESC["github_create_pull_request"],
    )
    async def github_create_pull_request(
        owner: str, repo: str, base: str, branch: str, title: str, body: str, files: dict[str, str]
    ) -> dict:
        """Open a DRAFT pull request that writes the given files on a new branch. Requires a human
        approval bound to this exact payload; without one it returns the approval command to run."""
        payload = {
            "owner": owner,
            "repo": repo,
            "base": base,
            "branch": branch,
            "title": title,
            "body": body,
            "files": files,
        }
        return await g.run(
            "github_create_pull_request",
            {"owner": owner, "repo": repo, "branch": branch},
            lambda: gh.create_pull_request(owner, repo, base, branch, title, body, files),
            approval_payload=payload,
        )

    return server


def main() -> None:
    root = Path(os.environ.get("ARP_HOME", ".arp"))
    principal = os.environ.get("ARP_PRINCIPAL", "anonymous")
    policy = Policy.load(os.environ.get("ARP_POLICY", "configs/policy.yaml"))
    audit = AuditLog(os.environ.get("ARP_AUDIT_LOG", root / "audit.jsonl"))
    approvals = ApprovalStore(os.environ.get("ARP_APPROVALS", root / "approvals"))
    pypi, gh = default_backends()
    build_server(principal, policy, audit, approvals, pypi, gh).run("stdio")


if __name__ == "__main__":
    main()
