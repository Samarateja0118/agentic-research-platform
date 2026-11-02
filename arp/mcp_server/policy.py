"""Permission checks, human approvals and the audit log for MCP tool calls.

The principal is fixed when the server process starts (``ARP_PRINCIPAL``). Agents cannot pick
their own identity per call: the orchestrator decides which principal each agent's MCP
session runs as, the same way a service account is bound to a deployment.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SECRET_KEYS = re.compile(r"(token|secret|password|authorization|api[_-]?key)", re.I)


@dataclass
class Decision:
    allowed: bool
    reason: str
    rule: str = ""


@dataclass
class PrincipalRule:
    tools: list[str]
    deny_tools: list[str] = field(default_factory=list)
    repos: list[str] = field(default_factory=list)  # owner/repo globs; empty = no repo access
    max_calls: int = 200  # per server process (one agent session)
    requires_approval: list[str] = field(default_factory=list)


class Policy:
    def __init__(self, principals: dict[str, PrincipalRule]):
        self.principals = principals

    @classmethod
    def load(cls, path: str | Path) -> Policy:
        raw = yaml.safe_load(Path(path).read_text())
        return cls({name: PrincipalRule(**rule) for name, rule in raw["principals"].items()})

    def check(self, principal: str, tool: str, args: dict[str, Any], calls_so_far: int) -> Decision:
        rule = self.principals.get(principal)
        if rule is None:
            return Decision(False, f"unknown principal '{principal}'", "default-deny")
        if any(fnmatch.fnmatch(tool, pat) for pat in rule.deny_tools):
            return Decision(False, f"tool '{tool}' is denied for '{principal}'", "deny_tools")
        if not any(fnmatch.fnmatch(tool, pat) for pat in rule.tools):
            return Decision(False, f"tool '{tool}' is not in the allowlist for '{principal}'", "tools")
        if calls_so_far >= rule.max_calls:
            return Decision(False, f"call budget of {rule.max_calls} exhausted", "max_calls")
        if "owner" in args and "repo" in args:
            full = f"{args['owner']}/{args['repo']}".lower()
            if not any(fnmatch.fnmatch(full, pat.lower()) for pat in rule.repos):
                return Decision(False, f"repo '{full}' is outside the allowlist for '{principal}'", "repos")
        return Decision(True, "allowed", "tools")

    def needs_approval(self, principal: str, tool: str) -> bool:
        rule = self.principals.get(principal)
        return bool(rule) and any(fnmatch.fnmatch(tool, p) for p in rule.requires_approval)


def payload_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class ApprovalStore:
    """Approvals are files a human writes with ``arp approve``. Each one binds to the digest of
    the exact payload, so an agent cannot get a patch approved and then push a different one."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def request(self, payload: dict[str, Any], summary: str) -> str:
        digest = payload_digest(payload)
        req_id = digest[:12]
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / f"{req_id}.request.json").write_text(
            json.dumps(
                {
                    "id": req_id,
                    "digest": digest,
                    "summary": summary,
                    "payload": payload,
                    "requested_at": time.time(),
                },
                indent=2,
            )
        )
        return req_id

    def approve(self, req_id: str, approver: str) -> None:
        req = json.loads((self.root / f"{req_id}.request.json").read_text())
        (self.root / f"{req_id}.approved.json").write_text(
            json.dumps(
                {"id": req_id, "digest": req["digest"], "approver": approver, "approved_at": time.time()},
                indent=2,
            )
        )

    def is_approved(self, payload: dict[str, Any]) -> tuple[bool, str]:
        digest = payload_digest(payload)
        path = self.root / f"{digest[:12]}.approved.json"
        if not path.exists():
            return False, f"no human approval for payload {digest[:12]}"
        rec = json.loads(path.read_text())
        if rec.get("digest") != digest:
            return False, "approval digest does not match payload"
        return True, f"approved by {rec.get('approver')}"


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: ("***" if SECRET_KEYS.search(k) else redact(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str) and len(value) > 300:
        return value[:300] + f"...[{len(value)} chars]"
    return value


class AuditLog:
    """Append-only JSONL. One line per tool call, including denied ones."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()

    def write(self, **record: Any) -> None:
        if self.path is None:
            return
        record = {"ts": round(time.time(), 3), "pid": os.getpid(), **record}
        if "args" in record:
            record["args"] = redact(record["args"])
        line = json.dumps(record, default=str)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as fh:
                fh.write(line + "\n")
