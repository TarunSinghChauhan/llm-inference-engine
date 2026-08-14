"""
speculative_compare.py
Benchmarks prompt lookup decoding (PLD) against standard autoregressive
decoding on the same model.

HONEST SCOPING NOTE: "speculative decoding" most commonly refers to using
a separate small draft model to propose tokens, verified by the large
target model. llama-cpp-python's high-level API does not cleanly expose
a dual-model draft/verify loop, so implementing that correctly would mean
hand-rolling token-level accept/reject logic against llama.cpp's internals
— a much larger undertaking than this project's scope justifies.

What's benchmarked here instead is prompt lookup decoding (PLD): rather
than a separate model, it searches the existing context for n-gram
matches and speculatively proposes the next few tokens from that match,
verifying them against the real model in one pass. It's a real, shipped
technique (also used in vLLM, HF transformers) and the speedup mechanism
is the same idea — propose cheaply, verify once — just without a second
model. It helps most on tasks with repetition (e.g. code, structured
output, or text that echoes parts of the prompt) and helps little to
none on free-form creative generation, which is itself a useful, honest
result to report.

Usage:
    python benchmarks/speculative_compare.py
"""

import time
import statistics
import json
from pathlib import Path

from llama_cpp import Llama
from llama_cpp.llama_speculative import LlamaPromptLookupDecoding

MODEL_PATH = Path(__file__).parent.parent / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf"
N_GPU_LAYERS = 0
N_CTX = 4096
MAX_TOKENS = 200

# Prompts chosen to span the range where PLD should and shouldn't help:
# repetitive/structured prompts (should benefit) vs open-ended prompts
# (unlikely to benefit much, since there's little to n-gram-match against).
PROMPTS = {
    "repetitive_structured": (
        "Here is a list of numbers: 1, 2, 3, 4, 5. "
        "Repeat this exact list back to me 5 times, once per line."
    ),
    "code_completion": (
        "Complete this Python function:\n"
        "def fibonacci(n):\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    return fibonacci(n-1) + fibonacci(n-2)\n\n"
        "Now write the equivalent function in JavaScript, matching the style exactly."
    ),
    "open_ended_creative": (
        "Write a short paragraph about the future of renewable energy."
    ),
}


def run_variant(use_pld: bool) -> dict:
    kwargs = dict(
        model_path=str(MODEL_PATH),
        n_gpu_layers=N_GPU_LAYERS,
        n_ctx=N_CTX,
        verbose=False,
    )
    if use_pld:
        # num_pred_tokens: how many tokens to speculatively propose per
        # lookup match before verifying against the real model.
        kwargs["draft_model"] = LlamaPromptLookupDecoding(num_pred_tokens=10)

    llm = Llama(**kwargs)

    results = {}
    for label, prompt in PROMPTS.items():
        latencies = []
        for _ in range(3):  # 3 runs per prompt to smooth noise
            start = time.time()
            out = llm(prompt, max_tokens=MAX_TOKENS, temperature=0.3)
            latencies.append((time.time() - start) * 1000)
        tokens = out["usage"]["completion_tokens"]
        p50 = statistics.median(latencies)
        results[label] = {
            "p50_ms": round(p50, 1),
            "tokens": tokens,
            "tokens_per_sec": round(tokens / (p50 / 1000), 2) if p50 > 0 else 0,
        }
        print(f"  [{'PLD' if use_pld else 'baseline'}] {label}: "
              f"p50={p50/1000:.1f}s, {results[label]['tokens_per_sec']} tok/s")

    del llm
    return results


if __name__ == "__main__":
    print("Running baseline (standard decoding) ...")
    baseline = run_variant(use_pld=False)

    print("\nRunning prompt lookup decoding ...")
    pld = run_variant(use_pld=True)

    print("\n" + "=" * 60)
    print("SUMMARY — tokens/sec (higher is better), speedup = PLD / baseline")
    print("=" * 60)
    for label in PROMPTS:
        b = baseline[label]["tokens_per_sec"]
        p = pld[label]["tokens_per_sec"]
        speedup = (p / b) if b > 0 else 0
        print(f"{label:24s}  baseline={b:>6.2f}  PLD={p:>6.2f}  speedup={speedup:.2f}x")

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"speculative_compare_{int(time.time())}.json"
    out_file.write_text(json.dumps({"baseline": baseline, "pld": pld}, indent=2))
    print(f"\nSaved: {out_file}")