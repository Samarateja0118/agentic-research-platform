"""Spawns the real MCP server over stdio against a small cassette (no network)."""

import json

from arp.agents.toolbox import MCPToolbox

CASSETTE = {
    "https://pypi.org/pypi/flask/json": {
        "status": 200,
        "body": {
            "info": {
                "name": "Flask",
                "version": "3.1.3",
                "summary": "web",
                "license": None,
                "license_expression": "BSD-3-Clause",
                "requires_python": ">=3.9",
                "requires_dist": ["click>=8.1.3", "pytest; extra == 'dev'"],
                "project_urls": {"Source": "https://github.com/pallets/flask"},
                "home_page": None,
            },
            "urls": [],
            "releases": {
                "3.1.3": [{"upload_time_iso_8601": "2026-01-01T00:00:00Z", "yanked": False}],
                "0.1": [{"upload_time_iso_8601": "2010-04-16T00:00:00Z", "yanked": False}],
            },
        },
    },
}


async def test_tools_policy_and_audit(tmp_path):
    (tmp_path / "cas").mkdir()
    (tmp_path / "cas" / "pypi.json").write_text(json.dumps(CASSETTE))
    env = {"ARP_HTTP_MODE": "replay", "ARP_CASSETTE_DIR": str(tmp_path / "cas"), "ARP_HOME": str(tmp_path)}
    async with MCPToolbox("researcher", env=env) as tb:
        assert {"pypi_package", "github_repo", "github_create_pull_request"} <= {t["name"] for t in tb.tools}
        text, err = await tb.call("w", "pypi_package", {"name": "flask"})
        data = json.loads(text)
        assert not err and data["latest_version"] == "3.1.3" and data["dependencies"] == ["click>=8.1.3"]
        assert data["first_release_date"] == "2010-04-16" and data["release_count"] == 2
        text, err = await tb.call("w", "pypi_package", {"name": "not-in-snapshot"})
        assert err and "not in snapshot" in text
        text, err = await tb.call("w", "github_repo", {"owner": "psf", "repo": "requests"})
        assert err and "permission denied" in text
    async with MCPToolbox("ops", env=env) as tb:
        payload = {
            "owner": "Samarateja0118",
            "repo": "agentic-research-platform",
            "base": "main",
            "branch": "b",
            "title": "t",
            "body": "b",
            "files": {"x.py": "y"},
        }
        text, err = await tb.call("ops", "github_create_pull_request", payload)
        assert err and "human approval required" in text
    audit = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text().splitlines()]
    assert [a["decision"] for a in audit] == ["allow", "allow", "deny", "pending_approval"]
    assert audit[1]["status"] == "error" and audit[0]["principal"] == "researcher"
