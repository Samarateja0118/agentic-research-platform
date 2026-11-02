import shutil
import subprocess

import pytest

from arp.sandbox.docker_sandbox import DockerSandbox, SandboxViolation, Workspace
from arp.sandbox.ops_agent import PROJECT, apply_bug

docker = pytest.mark.skipif(
    shutil.which("docker") is None
    or subprocess.run(["docker", "image", "inspect", "arp-sandbox:py311"], capture_output=True).returncode
    != 0,
    reason="needs docker and the arp-sandbox:py311 image",
)


def test_workspace_confines_paths_and_protects_tests():
    ws = Workspace(PROJECT)
    try:
        for bad in ("../../etc/passwd", "/etc/passwd", "ledger/../../x"):
            with pytest.raises(SandboxViolation):
                ws.resolve(bad)
        with pytest.raises(SandboxViolation):
            ws.write("tests/test_money.py", "def test_x(): pass")
        with pytest.raises(SandboxViolation):
            ws.write("ledger/new_file.py", "x = 1")
        ws.write("ledger/textutil.py", ws.read("ledger/textutil.py") + "\n# edit\n")
        assert "ledger/textutil.py" in ws.diff()
    finally:
        ws.cleanup()


def test_rejects_unsafe_test_target():
    ws = Workspace(PROJECT)
    try:
        sb = DockerSandbox(ws)
        import asyncio

        for bad in ("; rm -rf /", "../x", "ledger/money.py"):
            with pytest.raises(SandboxViolation):
                asyncio.run(sb.run_tests(bad))
    finally:
        ws.cleanup()


@docker
async def test_seeded_bug_fails_and_network_is_off():
    ws = Workspace(PROJECT)
    try:
        apply_bug(
            ws,
            {
                "id": "t",
                "file": "ledger/textutil.py",
                "edits": [
                    {"old": 'return text[: width - 3] + "..."', "new": 'return text[: width - 2] + "..."'}
                ],
            },
        )
        sb = DockerSandbox(ws, timeout=30)
        r = await sb.run_tests()
        assert r.exit_code == 1 and "test_truncate_exact_width" in r.output
        (ws.root / "tests" / "test_net.py").write_text(
            "import socket, pytest\n"
            "def test_no_network():\n"
            "    with pytest.raises(OSError):\n"
            "        socket.create_connection(('1.1.1.1', 53), timeout=3)\n"
            "def test_root_fs_read_only():\n"
            "    with pytest.raises(OSError):\n"
            "        open('/etc/evil', 'w')\n"
        )
        r = await sb.run_tests("tests/test_net.py")
        assert r.exit_code == 0, r.output
    finally:
        ws.cleanup()
