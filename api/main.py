"""
main.py
FastAPI entrypoint. Loads the model once at startup, exposes /generate
and /metrics. No external API calls — everything runs on local weights.
"""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel, Field

from engine.serving.model_loader import engine
from engine.serving.batcher import RequestBatcher
from api.queue import queue

# Micro-batching scheduler — see engine/serving/batcher.py module docstring
# for exactly what this does and does not give us on this CPU backend.
batcher = RequestBatcher(engine, max_batch_size=8, max_wait_ms=50)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load model + connect redis once
    engine.load()
    await queue.connect()
    batcher.start()
    yield
    # Shutdown
    await batcher.stop()


app = FastAPI(title="Local Inference Engine", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompts: list[str] = Field(..., description="One or more prompts — batched automatically")
    max_tokens: int = 256
    temperature: float = 0.7


class GenerateResponse(BaseModel):
    completions: list[str]
    latency_ms: float
    queue_depth_at_request: int


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    queue_depth = await queue.current_depth()
    req_id = await queue.track_start()

    # Each prompt is submitted independently to the batcher — if other
    # requests are in flight at the same moment, they'll be grouped into
    # the same dispatch window (see batcher.py).
    completions = await asyncio.gather(*[
        batcher.submit(p, max_tokens=req.max_tokens, temperature=req.temperature)
        for p in req.prompts
    ])
    completions = list(completions)

    latency_ms = await queue.track_end(req_id)

    return GenerateResponse(
        completions=completions,
        latency_ms=latency_ms,
        queue_depth_at_request=queue_depth,
    )


@app.get("/metrics")
async def metrics():
    """Feeds the dashboard — raw recent latencies for p50/p95/p99 calc."""
    latencies = await queue.recent_latencies()
    depth = await queue.current_depth()
    return {
        "queue_depth": depth,
        "sample_size": len(latencies),
        "recent_latencies_ms": latencies,
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model": engine.config.model_name}