"""
queue.py
Lightweight Redis-backed queue so we can observe queue depth / wait time
under load — this is what feeds the p50/p99 dashboard later.
"""

import time
import json
import uuid
import redis.asyncio as redis

REDIS_URL = "redis://localhost:6379/0"
QUEUE_KEY = "inference:queue_depth"


class RequestQueue:
    def __init__(self, redis_url: str = REDIS_URL):
        self._redis_url = redis_url
        self._client: redis.Redis | None = None

    async def connect(self):
        self._client = redis.from_url(self._redis_url, decode_responses=True)

    async def track_start(self) -> str:
        """Call when a request enters the handler. Returns a request id
        used to compute wait/serve time later."""
        req_id = str(uuid.uuid4())
        await self._client.incr(QUEUE_KEY)
        await self._client.hset(f"inference:req:{req_id}", mapping={
            "start_ts": time.time(),
        })
        return req_id

    async def track_end(self, req_id: str) -> float:
        """Call when generation completes. Returns total latency in ms
        and logs it to a rolling list Redis stores for the dashboard."""
        await self._client.decr(QUEUE_KEY)
        data = await self._client.hgetall(f"inference:req:{req_id}")
        start_ts = float(data.get("start_ts", time.time()))
        latency_ms = (time.time() - start_ts) * 1000

        await self._client.lpush("inference:latencies", json.dumps({
            "req_id": req_id,
            "latency_ms": latency_ms,
            "ts": time.time(),
        }))
        await self._client.ltrim("inference:latencies", 0, 9999)  # keep last 10k
        await self._client.delete(f"inference:req:{req_id}")
        return latency_ms

    async def current_depth(self) -> int:
        depth = await self._client.get(QUEUE_KEY)
        return int(depth) if depth else 0

    async def recent_latencies(self, n: int = 500) -> list[float]:
        raw = await self._client.lrange("inference:latencies", 0, n - 1)
        return [json.loads(r)["latency_ms"] for r in raw]


queue = RequestQueue()
