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


async def fire_request(client: httpx.AsyncClient, prompt: str) -> dict:
    start = time.time()
    resp = await client.post(
        "/generate",
        json={"prompts": [prompt], "max_tokens": 128},
        timeout=120.0,
    )
    wall_ms = (time.time() - start) * 1000
    data = resp.json()
    return {
        "wall_ms": wall_ms,
        "reported_latency_ms": data.get("latency_ms"),
        "status": resp.status_code,
    }


async def run(concurrency: int, total_requests: int, base_url: str):
    results = []
    async with httpx.AsyncClient(base_url=base_url) as client:
        sem = asyncio.Semaphore(concurrency)

        async def worker(i: int):
            async with sem:
                prompt = PROMPTS[i % len(PROMPTS)]
                r = await fire_request(client, prompt)
                results.append(r)

        await asyncio.gather(*[worker(i) for i in range(total_requests)])

    wall_times = sorted(r["wall_ms"] for r in results if r["status"] == 200)
    if not wall_times:
        print("No successful requests — is the server running?")
        return

    report = {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "successful": len(wall_times),
        "p50_ms": statistics.median(wall_times),
        "p95_ms": wall_times[int(len(wall_times) * 0.95) - 1],
        "p99_ms": wall_times[int(len(wall_times) * 0.99) - 1],
        "mean_ms": statistics.mean(wall_times),
        "throughput_req_per_sec": len(wall_times) / (sum(wall_times) / 1000 / concurrency),
    }

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
