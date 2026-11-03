"""Shows the human-approval gate on the one write tool, without pushing anything.

1. The ops principal asks to open a PR: refused, an approval request is filed.
2. The researcher principal asks for the same tool: denied by policy.
3. An approval for a *different* payload does not unlock this one.
Writes the audit lines to results/approval-gate-demo.txt.
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path

from arp.agents.toolbox import MCPToolbox
from arp.mcp_server.policy import ApprovalStore

PAYLOAD = {"owner": "Samarateja0118", "repo": "agentic-research-platform", "base": "main", "branch": "ops/fix-b08",
           "title": "fix(b08): truncate keeps width - 3 chars", "body": "demo",
           "files": {"evals/bugs/project/ledger/textutil.py": "fixed contents"}}


async def main() -> None:
    home = Path(tempfile.mkdtemp())
    env = {"ARP_HOME": str(home), "ARP_HTTP_MODE": "replay"}
    os.environ.update(env)
    out = []
    async with MCPToolbox("ops", env=env) as tb:
        text, err = await tb.call("ops", "github_create_pull_request", PAYLOAD)
        out.append(f"ops -> github_create_pull_request: error={err}: {text}")
    async with MCPToolbox("researcher", env=env) as tb:
        text, err = await tb.call("researcher", "github_create_pull_request", PAYLOAD)
        out.append(f"researcher -> github_create_pull_request: error={err}: {text}")
    store = ApprovalStore(home / "approvals")
    other = store.request({**PAYLOAD, "files": {"x.py": "something else"}}, "different patch")
    store.approve(other, "samara")
    async with MCPToolbox("ops", env=env) as tb:
        text, err = await tb.call("ops", "github_create_pull_request", PAYLOAD)
        out.append(f"ops, after approving a different patch: error={err}: {text}")
    out.append("\naudit log:")
    out += [json.dumps({k: v for k, v in json.loads(line).items() if k not in ("pid", "session")})
            for line in (home / "audit.jsonl").read_text().splitlines()]
    Path("results").mkdir(exist_ok=True)
    Path("results/approval-gate-demo.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out))


asyncio.run(main())
