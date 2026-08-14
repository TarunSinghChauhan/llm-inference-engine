"""
model_loader.py
Loads a local open-weights GGUF model via llama-cpp-python. No API keys,
no external calls. Sized for consumer GPUs (tested against a 4GB VRAM
RTX 2050) — partial GPU offload with CPU fallback for the rest of the
layers, so it runs even when the full model can't fit in VRAM.

Swap MODEL_PATH for any GGUF file you download (e.g. from Hugging Face's
GGUF repos for Qwen2.5-3B-Instruct or Llama-3.2-3B-Instruct).
"""

import os
from dataclasses import dataclass
from llama_cpp import Llama


@dataclass
class EngineConfig:
    # Point this at a local .gguf file. Recommended starting point:
    # Qwen2.5-3B-Instruct-Q4_K_M.gguf (~2GB) — fits comfortably in 4GB VRAM
    # with room for KV cache, and Q4_K_M keeps quality close to FP16.
    model_path: str = os.getenv("MODEL_PATH", "./models/qwen2.5-3b-instruct-q4_k_m.gguf")
    n_gpu_layers: int = int(os.getenv("N_GPU_LAYERS", "20"))  # tune down if you OOM; -1 = all layers on GPU
    n_ctx: int = int(os.getenv("MAX_MODEL_LEN", "4096"))
    n_batch: int = int(os.getenv("N_BATCH", "256"))  # prompt processing batch size
    n_threads: int = int(os.getenv("N_THREADS", os.cpu_count() or 4))
    verbose: bool = False


class InferenceEngine:
    """Thin wrapper around llama-cpp-python's Llama object so the rest of
    the codebase doesn't depend on llama.cpp's API directly — makes it
    easy to swap backends later (e.g. to vLLM on a bigger GPU) without
    touching callers."""

    def __init__(self, config: EngineConfig | None = None):
        self.config = config or EngineConfig()
        self._llm: Llama | None = None

    def load(self) -> "InferenceEngine":
        self._llm = Llama(
            model_path=self.config.model_path,
            n_gpu_layers=self.config.n_gpu_layers,
            n_ctx=self.config.n_ctx,
            n_batch=self.config.n_batch,
            n_threads=self.config.n_threads,
            verbose=self.config.verbose,
        )
        return self

    @property
    def llm(self) -> Llama:
        if self._llm is None:
            raise RuntimeError("Engine not loaded — call .load() first.")
        return self._llm

    def generate(self, prompts: list[str], max_tokens: int = 256, temperature: float = 0.7):
        """llama-cpp-python's core API is single-prompt. We loop here for
        now — this is exactly the naive baseline your Day 4-6 batching
        work replaces with a real request scheduler (engine/serving/batcher.py)
        that queues + groups concurrent requests instead of serializing them."""
        outputs = []
        for prompt in prompts:
            result = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            outputs.append(result["choices"][0]["text"])
        return outputs


# Singleton used by the FastAPI app so the model loads once at startup,
# not per-request.
engine = InferenceEngine()
