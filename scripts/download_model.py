"""
download_model.py
One-time helper to pull a quantized GGUF model from Hugging Face.
Run this before starting the server for the first time.

Usage:
    python scripts/download_model.py
"""

from pathlib import Path
from huggingface_hub import hf_hub_download

# Q4_K_M is the sweet spot for a 4GB card: ~2GB file, quality close to FP16.
REPO_ID = "Qwen/Qwen2.5-3B-Instruct-GGUF"
FILENAME = "qwen2.5-3b-instruct-q4_k_m.gguf"

if __name__ == "__main__":
    out_dir = Path(__file__).parent.parent / "models"
    out_dir.mkdir(exist_ok=True)

    print(f"Downloading {FILENAME} from {REPO_ID} ...")
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=str(out_dir),
    )
    print(f"Done: {path}")
    print("\nSet MODEL_PATH to this file (or leave the default in model_loader.py — it already points here).")
