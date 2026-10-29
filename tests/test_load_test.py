import importlib.util
from pathlib import Path

import httpx
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "run_load_test", Path(__file__).resolve().parents[1] / "benchmarks" / "run_load_test.py"
)
lt = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lt)


@pytest.mark.parametrize("n, pct, expected", [
    (100, 95, 95), (100, 99, 99), (100, 100, 100),
    (50, 99, 50),    # 49.5 rounds up to the slowest request, not the 49th
    (50, 95, 48),
    (10, 99, 10),
    (10, 95, 10),
    (1, 95, 1), (1, 50, 1),
])
def test_percentile_is_nearest_rank(n, pct, expected):
    assert lt.percentile(list(range(1, n + 1)), pct) == expected


def _client(handler):
    return httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_fire_request_reports_latency():
    async with _client(lambda r: httpx.Response(200, json={"latency_ms": 12.5})) as c:
        out = await lt.fire_request(c, "hi")
    assert out["status"] == 200
    assert out["reported_latency_ms"] == 12.5
    assert out["wall_ms"] >= 0


@pytest.mark.asyncio
async def test_fire_request_survives_a_non_json_error_page():
    async with _client(lambda r: httpx.Response(502, text="<html>Bad Gateway</html>")) as c:
        out = await lt.fire_request(c, "hi")
    assert out["status"] == 502
    assert out["reported_latency_ms"] is None


@pytest.mark.asyncio
async def test_fire_request_counts_a_connection_failure_instead_of_raising():
    def boom(request):
        raise httpx.ConnectError("refused")

    async with _client(boom) as c:
        out = await lt.fire_request(c, "hi")
    assert out["status"] == 0
    assert out["reported_latency_ms"] is None


def test_summarize_counts_failures_and_uses_elapsed_time():
    results = [{"wall_ms": float(ms), "status": 200} for ms in range(10, 110, 10)]
    results.append({"wall_ms": 5.0, "status": 0})
    report = lt.summarize(results, concurrency=4, elapsed_s=2.0)

    assert report["total_requests"] == 11
    assert report["successful"] == 10
    assert report["failed"] == 1
    assert report["p99_ms"] == 100.0  # the slowest successful request
    assert report["p50_ms"] == 55.0
    assert report["throughput_req_per_sec"] == 5.0  # 10 successes in 2s
