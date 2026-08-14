"""
download_quant_variants.py
Downloads additional GGUF quantization levels of the same base model so
we can benchmark latency/size/quality tradeoffs across them.

You already have Q4_K_M from download_model.py. This adds:
  - Q8_0    (larger, closer to FP16 quality — the "high quality" reference)
  - Q3_K_M  (smaller, more aggressive compression — the "fast/small" end)

Usage:
    python scripts/download_quant_variants.py
"""

from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "Qwen/Qwen2.5-3B-Instruct-GGUF"

VARIANTS = [
    "qwen2.5-3b-instruct-q8_0.gguf",
    "qwen2.5-3b-instruct-q3_k_m.gguf",
]

if __name__ == "__main__":
    out_dir = Path(__file__).parent.parent / "models"
    out_dir.mkdir(exist_ok=True)

    for filename in VARIANTS:
        print(f"Downloading {filename} ...")
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            local_dir=str(out_dir),
        )
        size_mb = Path(path).stat().st_size / (1024 * 1024)
        print(f"  Done: {path} ({size_mb:.0f} MB)\n")

    print("All variants downloaded. Run benchmarks/quantization_compare.py next.")