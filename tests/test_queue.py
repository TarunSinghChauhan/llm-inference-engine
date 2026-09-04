import json
import time
from unittest.mock import AsyncMock, patch

import pytest

from api.queue import RequestQueue


@pytest.mark.asyncio
async def test_track_start_increments_depth_and_stores_start_ts():
    q = RequestQueue()
    q._client = AsyncMock()
    req_id = await q.track_start()
    q._client.incr.assert_awaited_once_with("inference:queue_depth")
    args, kwargs = q._client.hset.call_args
    assert args[0] == f"inference:req:{req_id}"
    assert "start_ts" in kwargs["mapping"]


@pytest.mark.asyncio
async def test_track_end_computes_latency_and_cleans_up():
    q = RequestQueue()
    q._client = AsyncMock()
    start_ts = time.time() - 0.05  # ~50ms ago
    q._client.hgetall.return_value = {"start_ts": str(start_ts)}
    latency = await q.track_end("abc123")
    q._client.decr.assert_awaited_once_with("inference:queue_depth")
    assert latency >= 40
    q._client.delete.assert_awaited_once_with("inference:req:abc123")
    q._client.ltrim.assert_awaited_once_with("inference:latencies", 0, 9999)


@pytest.mark.asyncio
async def test_track_end_falls_back_to_now_when_start_ts_missing():
    q = RequestQueue()
    q._client = AsyncMock()
    q._client.hgetall.return_value = {}
    latency = await q.track_end("missing-id")
    assert latency < 50


@pytest.mark.asyncio
async def test_current_depth_returns_int_when_present():
    q = RequestQueue()
    q._client = AsyncMock()
    q._client.get.return_value = "7"
    assert await q.current_depth() == 7


@pytest.mark.asyncio
async def test_current_depth_returns_zero_when_none():
    q = RequestQueue()
    q._client = AsyncMock()
    q._client.get.return_value = None
    assert await q.current_depth() == 0


@pytest.mark.asyncio
async def test_recent_latencies_parses_json_entries():
    q = RequestQueue()
    q._client = AsyncMock()
    raw = [
        json.dumps({"req_id": "a", "latency_ms": 12.3, "ts": 1.0}),
        json.dumps({"req_id": "b", "latency_ms": 45.6, "ts": 2.0}),
    ]
    q._client.lrange.return_value = raw
    latencies = await q.recent_latencies(n=2)
    assert latencies == [12.3, 45.6]
    q._client.lrange.assert_awaited_once_with("inference:latencies", 0, 1)


@pytest.mark.asyncio
async def test_connect_creates_redis_client():
    q = RequestQueue()
    with patch("api.queue.redis.from_url") as mock_from_url:
        mock_from_url.return_value = AsyncMock()
        await q.connect()
        mock_from_url.assert_called_once_with(q._redis_url, decode_responses=True)
        assert q._client is not None
