"""
run_load_test.py
Simulates N concurrent requests against the running FastAPI server and
records latency distribution. Run this at concurrency=1 first to get
your baseline, then re-run at 10/50 to show batching's effect.

Usage:
    python benchmarks/run_load_test.py --concurrency 10 --requests 100
"""

import argparse
import asyncio
import math
import time
import json
import statistics
from pathlib import Path

import httpx

PROMPTS = [
    "Explain the difference between supervised and unsupervised learning.",
    "Write a Python function to reverse a linked list.",
    "Summarize the causes of the 2008 financial crisis in three sentences.",
    "What are the tradeoffs between REST and GraphQL?",
    "Describe how a transformer's attention mechanism works.",
]


def percentile(sorted_values: list, pct: float) -> float:
    """Nearest-rank percentile of an ascending list (pct in 0-100)."""
    rank = math.ceil(pct / 100 * len(sorted_values))
    return sorted_values[max(rank, 1) - 1]


async def fire_request(client: httpx.AsyncClient, prompt: str) -> dict:
    start = time.perf_counter()
    try:
        resp = await client.post(
            "/generate",
            json={"prompts": [prompt], "max_tokens": 128},
            timeout=120.0,
        )
    except httpx.HTTPError:
        # Timeout / connection failure: count it as a failed request instead of
        # letting one bad request abort the whole run.
        return {
            "wall_ms": (time.perf_counter() - start) * 1000,
            "reported_latency_ms": None,
            "status": 0,
        }
    wall_ms = (time.perf_counter() - start) * 1000
    try:
        reported = resp.json().get("latency_ms")
    except (ValueError, AttributeError):
        reported = None  # e.g. an HTML/plain-text 500 page
    return {
        "wall_ms": wall_ms,
        "reported_latency_ms": reported,
        "status": resp.status_code,
    }


def summarize(results: list, concurrency: int, elapsed_s: float) -> dict:
    wall_times = sorted(r["wall_ms"] for r in results if r["status"] == 200)
    return {
        "concurrency": concurrency,
        "total_requests": len(results),
        "successful": len(wall_times),
        "failed": len(results) - len(wall_times),
        "p50_ms": statistics.median(wall_times),
        "p95_ms": percentile(wall_times, 95),
        "p99_ms": percentile(wall_times, 99),
        "mean_ms": statistics.mean(wall_times),
        "throughput_req_per_sec": len(wall_times) / elapsed_s,
    }


async def run(concurrency: int, total_requests: int, base_url: str):
    results = []
    started = time.perf_counter()
    async with httpx.AsyncClient(base_url=base_url) as client:
        sem = asyncio.Semaphore(concurrency)

        async def worker(i: int):
            async with sem:
                prompt = PROMPTS[i % len(PROMPTS)]
                r = await fire_request(client, prompt)
                results.append(r)

        await asyncio.gather(*[worker(i) for i in range(total_requests)])
    elapsed_s = time.perf_counter() - started

    if not any(r["status"] == 200 for r in results):
        print("No successful requests — is the server running?")
        return

    report = summarize(results, concurrency, elapsed_s)

    print(json.dumps(report, indent=2))

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"loadtest_c{concurrency}_{int(time.time())}.json"
    out_file.write_text(json.dumps({"report": report, "raw": results}, indent=2))
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--base-url", type=str, default="http://localhost:8000")
    args = parser.parse_args()

    asyncio.run(run(args.concurrency, args.requests, args.base_url))
