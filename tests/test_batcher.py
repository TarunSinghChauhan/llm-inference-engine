import asyncio
import pytest

from engine.serving.batcher import RequestBatcher


class FakeEngine:
    def __init__(self, responses=None, raise_error=None):
        self.responses = responses
        self.raise_error = raise_error
        self.calls = []

    def generate(self, prompts, max_tokens, temperature):
        self.calls.append(list(prompts))
        if self.raise_error:
            raise self.raise_error
        return self.responses or [f"response-to-{p}" for p in prompts]


@pytest.mark.asyncio
async def test_batch_dispatches_when_max_batch_size_reached():
    engine = FakeEngine()
    batcher = RequestBatcher(engine, max_batch_size=2, max_wait_ms=5000)
    batcher.start()

    results = await asyncio.gather(
        batcher.submit("prompt-a"),
        batcher.submit("prompt-b"),
    )

    await batcher.stop()
    assert results == ["response-to-prompt-a", "response-to-prompt-b"]
    assert len(engine.calls) == 1
    assert set(engine.calls[0]) == {"prompt-a", "prompt-b"}


@pytest.mark.asyncio
async def test_batch_dispatches_after_timeout_with_partial_batch():
    engine = FakeEngine()
    batcher = RequestBatcher(engine, max_batch_size=10, max_wait_ms=50)
    batcher.start()

    result = await batcher.submit("solo-prompt")

    await batcher.stop()
    assert result == "response-to-solo-prompt"
    assert len(engine.calls) == 1
    assert engine.calls[0] == ["solo-prompt"]


@pytest.mark.asyncio
async def test_engine_exception_propagates_to_all_requests_in_batch():
    engine = FakeEngine(raise_error=RuntimeError("engine crashed"))
    batcher = RequestBatcher(engine, max_batch_size=2, max_wait_ms=5000)
    batcher.start()

    async def submit_and_catch(prompt):
        try:
            await batcher.submit(prompt)
            return None
        except RuntimeError as e:
            return str(e)

    results = await asyncio.gather(
        submit_and_catch("prompt-x"),
        submit_and_catch("prompt-y"),
    )

    await batcher.stop()
    assert results == ["engine crashed", "engine crashed"]


@pytest.mark.asyncio
async def test_multiple_batches_processed_independently():
    engine = FakeEngine()
    batcher = RequestBatcher(engine, max_batch_size=1, max_wait_ms=5000)
    batcher.start()

    r1 = await batcher.submit("first")
    r2 = await batcher.submit("second")

    await batcher.stop()
    assert r1 == "response-to-first"
    assert r2 == "response-to-second"
    assert len(engine.calls) == 2
