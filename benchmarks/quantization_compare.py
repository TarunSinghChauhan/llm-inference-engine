"""
quantization_compare.py
Loads each downloaded GGUF quantization variant one at a time, runs an
identical prompt set through each, and records:
  - latency per request (p50/mean)
  - tokens/sec
  - model file size on disk (memory footprint proxy)
  - raw outputs, saved so you can manually compare quality side by side

This does NOT compute a formal perplexity/accuracy score — that requires
a held-out eval corpus and llama.cpp's perplexity tool, which is a
separate, heavier undertaking. What this gives you is a fair, honest
latency/size comparison plus raw outputs for a qualitative quality check
you can describe accurately in your README (e.g. "outputs from Q3_K_M
showed occasional grammatical slips vs Q8_0 on complex prompts" — written
from what you actually observe, not fabricated).

Usage:
    python benchmarks/quantization_compare.py
"""

import json
import time
import statistics
from pathlib import Path

from llama_cpp import Llama

MODELS_DIR = Path(__file__).parent.parent / "models"

VARIANTS = {
    "Q3_K_M": "qwen2.5-3b-instruct-q3_k_m.gguf",
    "Q4_K_M": "qwen2.5-3b-instruct-q4_k_m.gguf",
    "Q8_0":   "qwen2.5-3b-instruct-q8_0.gguf",
}

PROMPTS = [
    "Explain the difference between supervised and unsupervised learning.",
    "Write a Python function to reverse a linked list.",
    "What are the tradeoffs between REST and GraphQL?",
    "Describe how a transformer's attention mechanism works.",
    "Summarize the causes of the 2008 financial crisis in three sentences.",
]

N_GPU_LAYERS = 0  # set to your working value if you have GPU offload working
N_CTX = 4096
MAX_TOKENS = 128


def benchmark_variant(name: str, filename: str) -> dict:
    model_path = MODELS_DIR / filename
    if not model_path.exists():
        print(f"  Skipping {name} — file not found: {model_path}")
        return None

    file_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"\nLoading {name} ({file_size_mb:.0f} MB) ...")

    llm = Llama(
        model_path=str(model_path),
        n_gpu_layers=N_GPU_LAYERS,
        n_ctx=N_CTX,
        verbose=False,
    )

    latencies_ms = []
    total_tokens = 0
    outputs = []

    for prompt in PROMPTS:
        start = time.time()
        result = llm(prompt, max_tokens=MAX_TOKENS, temperature=0.7)
        elapsed_ms = (time.time() - start) * 1000

        text = result["choices"][0]["text"]
        completion_tokens = result["usage"]["completion_tokens"]

        latencies_ms.append(elapsed_ms)
        total_tokens += completion_tokens
        outputs.append({"prompt": prompt, "output": text.strip()})

        print(f"  '{prompt[:40]}...' -> {elapsed_ms/1000:.1f}s, {completion_tokens} tokens")

    total_time_s = sum(latencies_ms) / 1000
    tokens_per_sec = total_tokens / total_time_s if total_time_s > 0 else 0

    del llm  # free memory before loading next variant

    return {
        "variant": name,
        "file_size_mb": round(file_size_mb, 1),
        "p50_latency_ms": round(statistics.median(latencies_ms), 1),
        "mean_latency_ms": round(statistics.mean(latencies_ms), 1),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "outputs": outputs,
    }


if __name__ == "__main__":
    results = []
    for name, filename in VARIANTS.items():
        r = benchmark_variant(name, filename)
        if r:
            results.append(r)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"{r['variant']:8s}  size={r['file_size_mb']:>7.1f}MB  "
              f"p50={r['p50_latency_ms']/1000:>5.1f}s  "
              f"tok/s={r['tokens_per_sec']:>5.2f}")

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"quant_compare_{int(time.time())}.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nFull results (including raw outputs for quality comparison) saved to:\n{out_file}")