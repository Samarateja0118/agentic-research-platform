"""A throwaway workspace plus a locked-down container for running its tests.

File tools act on a host-side copy of the repo, confined to that directory. Code from the repo
only ever executes inside a container with no network, a read-only root filesystem, dropped
capabilities, a non-root user, and CPU, memory, process and wall-clock limits.
"""

from __future__ import annotations

import asyncio
import difflib
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

IMAGE = os.environ.get("ARP_SANDBOX_IMAGE", "arp-sandbox:py311")
SAFE_TARGET = re.compile(r"^[\w./\-:\[\]]*$")


class SandboxViolation(Exception):
    pass


@dataclass
class RunResult:
    exit_code: int
    output: str
    timed_out: bool


class Workspace:
    def __init__(self, source: str | Path, readonly_globs: tuple[str, ...] = ("tests/*",)):
        self.source = Path(source)
        self.root = Path(tempfile.mkdtemp(prefix="arp-ws-")) / "repo"
        shutil.copytree(self.source, self.root, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
        self.original = {p: self.read(p) for p in self.files()}
        self.readonly_globs = readonly_globs

    def resolve(self, rel: str) -> Path:
        if rel.startswith("/") or "\x00" in rel:
            raise SandboxViolation(f"absolute or invalid path: {rel!r}")
        p = (self.root / rel).resolve()
        if p != self.root.resolve() and self.root.resolve() not in p.parents:
            raise SandboxViolation(f"path escapes workspace: {rel!r}")
        return p

    def files(self) -> list[str]:
        return sorted(
            str(p.relative_to(self.root))
            for p in self.root.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        )

    def read(self, rel: str) -> str:
        return self.resolve(rel).read_text()

    def is_readonly(self, rel: str) -> bool:
        from fnmatch import fnmatch

        return any(fnmatch(rel, g) for g in self.readonly_globs)

    def write(self, rel: str, text: str) -> None:
        if self.is_readonly(rel):
            raise SandboxViolation(f"{rel} is read-only for the agent (tests may not be edited)")
        p = self.resolve(rel)
        if not p.exists():
            raise SandboxViolation(f"{rel} does not exist; the agent may only edit existing files")
        p.write_text(text)

    def diff(self) -> str:
        out = []
        for rel in self.files():
            before, after = self.original.get(rel, ""), self.read(rel)
            if before != after:
                out += difflib.unified_diff(
                    before.splitlines(True), after.splitlines(True), f"a/{rel}", f"b/{rel}"
                )
        return "".join(out)

    def changed_files(self) -> dict[str, str]:
        return {rel: self.read(rel) for rel in self.files() if self.original.get(rel) != self.read(rel)}

    def cleanup(self) -> None:
        shutil.rmtree(self.root.parent, ignore_errors=True)


class DockerSandbox:
    def __init__(
        self,
        workspace: Workspace,
        image: str = IMAGE,
        timeout: float = 60.0,
        memory: str = "512m",
        cpus: str = "1",
    ):
        self.ws, self.image, self.timeout, self.memory, self.cpus = workspace, image, timeout, memory, cpus

    def command(self, argv: list[str]) -> list[str]:
        os.chmod(self.ws.root, 0o777)
        return [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,size=64m",
            "--memory",
            self.memory,
            "--cpus",
            self.cpus,
            "--pids-limit",
            "128",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            "1000:1000",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            "-v",
            f"{self.ws.root}:/work:rw",
            "-w",
            "/work",
            self.image,
            "timeout",
            str(int(self.timeout)),
            *argv,
        ]

    async def run_tests(self, target: str = "") -> RunResult:
        target = target.strip()
        if target and (not SAFE_TARGET.match(target) or not target.startswith("tests")):
            raise SandboxViolation(f"test target must be a path under tests/: {target!r}")
        argv = ["python", "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"] + (
            [target] if target else []
        )
        proc = await asyncio.create_subprocess_exec(
            *self.command(argv), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), self.timeout + 30)
        except TimeoutError:
            proc.kill()
            return RunResult(124, "sandbox wall-clock timeout", True)
        return RunResult(proc.returncode or 0, out.decode(errors="replace"), proc.returncode == 124)
