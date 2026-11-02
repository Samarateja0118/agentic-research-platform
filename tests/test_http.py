import httpx
import pytest

from arp.mcp_server.http import Cassette, CassetteMiss, HttpBackend, RetryPolicy, UpstreamError


def backend(handler, mode="live", cassette=None):
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)

    b = HttpBackend(
        "t",
        mode,
        cassette,
        transport=httpx.MockTransport(handler),
        sleep=fake_sleep,
        retry=RetryPolicy(max_attempts=4, base_delay=0.1),
    )
    return b, sleeps


async def test_retries_5xx_then_succeeds():
    calls = []

    def h(req):
        calls.append(1)
        return httpx.Response(503) if len(calls) < 3 else httpx.Response(200, json={"ok": True})

    b, sleeps = backend(h)
    r = await b.get_json("https://x/y")
    assert r.body == {"ok": True} and r.attempts == 3 and len(sleeps) == 2


async def test_honours_retry_after():
    calls = []

    def h(req):
        calls.append(1)
        return (
            httpx.Response(429, headers={"retry-after": "2"})
            if len(calls) == 1
            else httpx.Response(200, json={})
        )

    b, sleeps = backend(h)
    await b.get_json("https://x/y")
    assert sleeps == [2.0]


async def test_no_retry_on_404():
    calls = []

    def h(req):
        calls.append(1)
        return httpx.Response(404, json={"message": "nope"})

    b, _ = backend(h)
    r = await b.get_json("https://x/y")
    assert r.status == 404 and len(calls) == 1


async def test_gives_up_after_max_attempts():
    b, sleeps = backend(lambda req: httpx.Response(502))
    with pytest.raises(UpstreamError) as e:
        await b.get_json("https://x/y")
    assert e.value.attempts == 4 and len(sleeps) == 3


async def test_timeouts_are_retried():
    calls = []

    def h(req):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectTimeout("slow")
        return httpx.Response(200, json={"v": 1})

    b, _ = backend(h)
    assert (await b.get_json("https://x/y")).body == {"v": 1}


async def test_record_then_replay(tmp_path):
    cas = Cassette(tmp_path / "c.json")
    b, _ = backend(lambda req: httpx.Response(200, json={"n": 1}), "record", cas)
    await b.get_json("https://x/a")
    offline, _ = backend(
        lambda req: (_ for _ in ()).throw(AssertionError("network used")),
        "replay",
        Cassette(tmp_path / "c.json"),
    )
    assert (await offline.get_json("https://x/a")).from_cassette
    with pytest.raises(CassetteMiss):
        await offline.get_json("https://x/b")


async def test_writes_refused_in_replay(tmp_path):
    b, _ = backend(lambda req: httpx.Response(200), "replay", Cassette(tmp_path / "c.json"))
    with pytest.raises(UpstreamError):
        await b.send_json("POST", "https://x/a", {})
