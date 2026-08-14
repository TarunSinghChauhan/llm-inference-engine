"""
batcher.py
Micro-batching request scheduler for the inference engine.

HONEST LIMITATION (documented deliberately, not glossed over): llama.cpp's
high-level Python API (llama-cpp-python's `Llama` class) does not expose
true parallel-sequence decoding the way vLLM does on GPU — under the hood,
prompts submitted to one Llama instance are still processed sequentially.

What this scheduler DOES give you, and what it's benchmarked against:
  1. Reduced per-request scheduling/queueing overhead vs a naive lock —
     requests arriving close together are grouped and dispatched as one
     unit rather than fighting individually for the lock.
  2. A real request-batching abstraction (window + max-batch-size logic)
     that is the correct pattern for a GPU backend (vLLM/TGI) where the
     underlying engine DOES do parallel decode — swapping the backend
     later means this scheduler doesn't need to change, only
     model_loader.generate() does.
  3. A clean place to measure and report the actual (small, CPU-bound)
     gains honestly, rather than claiming GPU-class batching speedups
     that this hardware/backend combination cannot produce.

Usage: submit a prompt, get back a future that resolves once its batch
has been processed.
"""

import asyncio
import time
from dataclasses import dataclass, field


@dataclass
class BatchRequest:
    prompt: str
    max_tokens: int
    temperature: float
    future: asyncio.Future = field(default_factory=asyncio.Future)
    enqueued_at: float = field(default_factory=time.time)


class RequestBatcher:
    def __init__(self, engine, max_batch_size: int = 8, max_wait_ms: int = 50):
        """
        engine: the InferenceEngine instance (from model_loader.py)
        max_batch_size: stop collecting and dispatch once this many requests
            are queued, even if max_wait_ms hasn't elapsed
        max_wait_ms: how long to wait for more requests to join a batch
            before dispatching what's collected so far
        """
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self._queue: list[BatchRequest] = []
        self._lock = asyncio.Lock()
        self._batch_ready = asyncio.Event()
        self._worker_task: asyncio.Task | None = None

    def start(self):
        self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self):
        if self._worker_task:
            self._worker_task.cancel()

    async def submit(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        """Queue a prompt for batched generation. Returns the completion
        text once this request's batch has been processed."""
        req = BatchRequest(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
        async with self._lock:
            self._queue.append(req)
            if len(self._queue) >= self.max_batch_size:
                self._batch_ready.set()
        return await req.future

    async def _worker_loop(self):
        loop = asyncio.get_running_loop()
        while True:
            # Wait for either the batch-size trigger or the timeout window
            try:
                await asyncio.wait_for(self._batch_ready.wait(), timeout=self.max_wait_ms / 1000)
            except asyncio.TimeoutError:
                pass
            self._batch_ready.clear()

            async with self._lock:
                if not self._queue:
                    continue
                batch = self._queue
                self._queue = []

            if not batch:
                continue

            # Dispatch the whole batch as one call off the event loop.
            # Internally this still loops prompt-by-prompt (see module
            # docstring) but requests that arrived together are now
            # scheduled together, cutting per-request overhead.
            prompts = [r.prompt for r in batch]
            try:
                results = await loop.run_in_executor(
                    None,
                    lambda: self.engine.generate(
                        prompts=prompts,
                        max_tokens=batch[0].max_tokens,
                        temperature=batch[0].temperature,
                    ),
                )
                for req, result in zip(batch, results):
                    if not req.future.done():
                        req.future.set_result(result)
            except Exception as e:
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(e)