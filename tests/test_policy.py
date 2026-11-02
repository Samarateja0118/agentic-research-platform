import json

from arp.mcp_server.policy import ApprovalStore, AuditLog, Policy, PrincipalRule

POLICY = Policy(
    {
        "researcher": PrincipalRule(
            tools=["pypi_*", "github_*"],
            deny_tools=["github_create_pull_request"],
            repos=["samarateja0118/*"],
            max_calls=2,
        ),
        "ops": PrincipalRule(
            tools=["github_create_pull_request"],
            repos=["samarateja0118/agentic-research-platform"],
            requires_approval=["github_create_pull_request"],
        ),
    }
)


def test_unknown_principal_denied():
    assert not POLICY.check("intruder", "pypi_package", {}, 0).allowed


def test_tool_allowlist_and_denylist():
    assert POLICY.check("researcher", "pypi_package", {"name": "x"}, 0).allowed
    assert not POLICY.check("researcher", "github_create_pull_request", {}, 0).allowed
    assert not POLICY.check("ops", "pypi_package", {}, 0).allowed


def test_repo_allowlist_case_insensitive():
    assert POLICY.check("researcher", "github_repo", {"owner": "Samarateja0118", "repo": "x"}, 0).allowed
    d = POLICY.check("researcher", "github_repo", {"owner": "psf", "repo": "requests"}, 0)
    assert not d.allowed and d.rule == "repos"


def test_call_budget():
    assert not POLICY.check("researcher", "pypi_package", {}, 2).allowed


def test_approval_binds_to_exact_payload(tmp_path):
    store = ApprovalStore(tmp_path)
    payload = {"files": {"a.py": "fixed"}, "title": "fix"}
    rid = store.request(payload, "ops -> pr")
    assert not store.is_approved(payload)[0]
    store.approve(rid, "samara")
    assert store.is_approved(payload)[0]
    assert not store.is_approved({**payload, "files": {"a.py": "something else"}})[0]


def test_audit_redacts_secrets(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl")
    log.write(tool="t", args={"token": "abc", "name": "x", "body": "y" * 1000})
    rec = json.loads((tmp_path / "a.jsonl").read_text())
    assert rec["args"]["token"] == "***" and rec["args"]["name"] == "x" and len(rec["args"]["body"]) < 400
