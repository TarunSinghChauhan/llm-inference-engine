# Local LLM Inference Optimization Engine

A serving layer for open-weights LLMs focused on throughput and latency —
continuous batching, KV-cache reuse, quantization tradeoffs, and speculative
decoding. Runs fully locally on open model weights. No API keys, no per-token
billing.

Backend: `llama.cpp` via `llama-cpp-python`, chosen to run on consumer GPUs
with limited VRAM (developed and benchmarked on an RTX 2050, 4GB VRAM) —
partial GPU offload with CPU fallback, GGUF quantization native.

## Results

*(fill in after Day 1–9 benchmarks — this table is what goes in outreach emails)*

| Config              | p50 latency | p99 latency | Throughput (req/s) |
|---------------------|-------------|-------------|---------------------|
| Single request (FP16, no batching) | — | — | — |
| Continuous batching (c=10)         | — | — | — |
| Continuous batching (c=50)         | — | — | — |
| INT8 quantized                     | — | — | — |
| GGUF Q4 quantized                  | — | — | — |
| + Speculative decoding             | — | — | — |

## Quickstart (Windows, native — recommended for development)

Docker GPU passthrough on Windows adds friction for no benefit during dev.
Run natively; use Docker only for the final "recruiter can clone and run
it" packaging (Day 12–13).

```powershell
# 1. Install llama-cpp-python with CUDA support (needs CUDA toolkit installed —
#    check with `nvcc --version`; if missing, install from developer.nvidia.com)
$env:CMAKE_ARGS="-DGGML_CUDA=on"
pip install -r requirements.txt

# 2. Download the model (one-time, ~2GB)
python scripts/download_model.py

# 3. Start Redis (via Docker is fine — it's lightweight, no GPU needed)
docker run -d -p 6379:6379 redis:7-alpine

# 4. Start the server
uvicorn api.main:app --reload
```

Then hit it:

```powershell
curl -X POST http://localhost:8000/generate `
  -H "Content-Type: application/json" `
  -d '{\"prompts\": [\"Explain gradient descent in one paragraph.\"], \"max_tokens\": 128}'
```

If CUDA build tools aren't set up and you just want it running first,
skip the `$env:CMAKE_ARGS` line — `llama-cpp-python` installs CPU-only
and still works, just slower. Set `N_GPU_LAYERS=0` in that case.

**VRAM tip:** if you get a CUDA out-of-memory error, lower `N_GPU_LAYERS`
in your environment (start at 20, drop to 12–15 on a 4GB card if needed).
This tradeoff — GPU layers vs VRAM headroom — is itself a benchmarkable
result worth including in your write-up.

## Baseline benchmark

```bash
python benchmarks/run_load_test.py --concurrency 1 --requests 50
python benchmarks/run_load_test.py --concurrency 10 --requests 200
python benchmarks/run_load_test.py --concurrency 50 --requests 500
```

Results are saved to `benchmarks/results/`.

## Architecture

- `engine/serving/` — model loading + vLLM wrapper (prefix caching enabled by default for KV-cache reuse)
- `api/` — FastAPI app + Redis-backed request queue for latency tracking
- `benchmarks/` — load test runner, produces the numbers in the table above
- `dashboard/` — live p50/p99 view (Day 12–13)

## Roadmap

- [x] Baseline FastAPI + vLLM serving
- [ ] Continuous batching load test (c=1/10/50)
- [ ] INT8 / GGUF Q4 quantization + benchmark
- [ ] Speculative decoding (draft: Qwen2.5-0.5B, target: Qwen2.5-3B)
- [ ] Live latency dashboard
