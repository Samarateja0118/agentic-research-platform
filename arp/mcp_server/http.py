"""Outbound HTTP for MCP backends: retries with jittered backoff, Retry-After, and a cassette.

The cassette makes evals reproducible. In ``record`` mode every GET is fetched live and stored.
In ``replay`` mode nothing touches the network and a miss is an error. ``replay_or_record``
serves stored responses and records anything new, which is how the eval snapshot was built.
"""

from __future__ import annotations

import asyncio
import email.utils
import hashlib
import json
import os
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

RETRY_STATUSES = {408, 425, 429, 500, 502, 503, 504}


class UpstreamError(Exception):
    def __init__(self, message: str, status: int | None = None, attempts: int = 1):
        super().__init__(message)
        self.status = status
        self.attempts = attempts


class CassetteMiss(UpstreamError):
    pass


@dataclass
class FetchResult:
    status: int
    body: Any
    attempts: int
    from_cassette: bool
    latency_ms: float


class Cassette:
    """JSON file of url -> {status, body}. One file per backend keeps diffs readable."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._data: dict[str, dict] = {}
        if path.exists():
            self._data = json.loads(path.read_text())

    @staticmethod
    def key(url: str) -> str:
        return url

    def get(self, url: str) -> dict | None:
        return self._data.get(self.key(url))

    def put(self, url: str, status: int, body: Any) -> None:
        with self._lock:
            self._data[self.key(url)] = {"status": status, "body": body}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._data, indent=0, sort_keys=True))
            tmp.replace(self.path)

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self._data, sort_keys=True).encode()).hexdigest()[:12]


def _retry_after_seconds(resp: httpx.Response) -> float | None:
    value = resp.headers.get("retry-after")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        parsed = email.utils.parsedate_to_datetime(value)
        return max(0.0, parsed.timestamp() - time.time())


@dataclass
class RetryPolicy:
    max_attempts: int = 4
    base_delay: float = 0.5
    max_delay: float = 8.0
    max_retry_after: float = 30.0

    def backoff(self, attempt: int) -> float:
        # full jitter: uniform(0, min(cap, base * 2^attempt))
        return random.uniform(0, min(self.max_delay, self.base_delay * (2**attempt)))


@dataclass
class HttpBackend:
    name: str
    mode: str = "live"  # live | record | replay | replay_or_record
    cassette: Cassette | None = None
    headers: dict[str, str] = field(default_factory=dict)
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    timeout: float = 15.0
    transport: httpx.AsyncBaseTransport | None = None  # tests inject httpx.MockTransport
    sleep: Any = asyncio.sleep
    shrink: Any = None  # optional (url, body) -> body, applied before caching and returning

    def __post_init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers=self.headers, timeout=self.timeout, transport=self.transport, follow_redirects=True
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    async def get_json(self, url: str) -> FetchResult:
        start = time.perf_counter()
        if self.mode in ("replay", "replay_or_record") and self.cassette is not None:
            hit = self.cassette.get(url)
            if hit is not None:
                return FetchResult(hit["status"], hit["body"], 0, True, (time.perf_counter() - start) * 1e3)
            if self.mode == "replay":
                raise CassetteMiss(f"{self.name}: not in snapshot: {url}", status=None, attempts=0)
        status, body, attempts = await self._fetch_with_retries("GET", url)
        if self.shrink is not None and status < 400:
            body = self.shrink(url, body)
        if self.mode in ("record", "replay_or_record") and self.cassette is not None and status < 500:
            self.cassette.put(url, status, body)
        return FetchResult(status, body, attempts, False, (time.perf_counter() - start) * 1e3)

    async def send_json(self, method: str, url: str, payload: dict) -> FetchResult:
        """Writes are never recorded or replayed and never retried on 4xx."""
        if self.mode == "replay":
            raise UpstreamError(f"{self.name}: writes are disabled in replay mode")
        start = time.perf_counter()
        status, body, attempts = await self._fetch_with_retries(method, url, payload)
        return FetchResult(status, body, attempts, False, (time.perf_counter() - start) * 1e3)

    async def _fetch_with_retries(self, method: str, url: str, payload: dict | None = None):
        last_exc: Exception | None = None
        for attempt in range(self.retry.max_attempts):
            try:
                resp = await self._http().request(method, url, json=payload)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                if attempt + 1 < self.retry.max_attempts:
                    await self.sleep(self.retry.backoff(attempt))
                continue
            if resp.status_code in RETRY_STATUSES and attempt + 1 < self.retry.max_attempts:
                delay = _retry_after_seconds(resp)
                if delay is None or delay > self.retry.max_retry_after:
                    delay = self.retry.backoff(attempt)
                await self.sleep(delay)
                continue
            try:
                body = resp.json()
            except ValueError:
                body = {"text": resp.text[:2000]}
            if resp.status_code >= 500 or resp.status_code in RETRY_STATUSES:
                raise UpstreamError(
                    f"{self.name}: HTTP {resp.status_code} after {attempt + 1} attempts",
                    resp.status_code,
                    attempt + 1,
                )
            return resp.status_code, body, attempt + 1
        raise UpstreamError(
            f"{self.name}: {type(last_exc).__name__} after {self.retry.max_attempts} attempts",
            None,
            self.retry.max_attempts,
        )


def backend_mode() -> str:
    return os.environ.get("ARP_HTTP_MODE", "replay_or_record")
