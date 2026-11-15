import asyncio

import pytest

from engine.serving.batcher import RequestBatcher


class RecordingEngine:
    def __init__(self, fail_when=None, short=False):
        self.calls = []  # (prompts, max_tokens, temperature)
        self.fail_when = fail_when
        self.short = short

    def generate(self, prompts, max_tokens, temperature):
        self.calls.append((list(prompts), max_tokens, temperature))
        if self.fail_when and self.fail_when(max_tokens, temperature):
            raise RuntimeError("boom")
        out = [f"{p}|{max_tokens}|{temperature}" for p in prompts]
        return out[:-1] if self.short else out


@pytest.mark.asyncio
async def test_each_request_is_generated_with_its_own_max_tokens_and_temperature():
    engine = RecordingEngine()
    batcher = RequestBatcher(engine, max_batch_size=3, max_wait_ms=5000)
    batcher.start()

    results = await asyncio.wait_for(asyncio.gather(
        batcher.submit("a", max_tokens=10, temperature=0.0),
        batcher.submit("b", max_tokens=256, temperature=0.7),
        batcher.submit("c", max_tokens=10, temperature=0.0),
    ), timeout=2)
    await batcher.stop()

    # nobody gets someone else's settings, and results map back to the right caller
    assert results == ["a|10|0.0", "b|256|0.7", "c|10|0.0"]
    assert sorted(engine.calls) == [(["a", "c"], 10, 0.0), (["b"], 256, 0.7)]


@pytest.mark.asyncio
async def test_requests_with_identical_settings_still_share_one_engine_call():
    engine = RecordingEngine()
    batcher = RequestBatcher(engine, max_batch_size=3, max_wait_ms=5000)
    batcher.start()
    await asyncio.wait_for(asyncio.gather(*[batcher.submit(p) for p in "xyz"]), timeout=2)
    await batcher.stop()
    assert len(engine.calls) == 1
    assert sorted(engine.calls[0][0]) == ["x", "y", "z"]


@pytest.mark.asyncio
async def test_a_burst_is_split_into_batches_of_at_most_max_batch_size():
    engine = RecordingEngine()
    batcher = RequestBatcher(engine, max_batch_size=2, max_wait_ms=5000)
    batcher.start()

    prompts = [f"p{i}" for i in range(5)]
    results = await asyncio.wait_for(
        asyncio.gather(*[batcher.submit(p) for p in prompts]), timeout=2
    )
    await batcher.stop()

    assert results == [f"{p}|256|0.7" for p in prompts]
    assert all(len(c[0]) <= 2 for c in engine.calls)
    assert sum(len(c[0]) for c in engine.calls) == 5


@pytest.mark.asyncio
async def test_one_failing_settings_group_does_not_fail_the_others():
    engine = RecordingEngine(fail_when=lambda m, t: t == 0.0)
    batcher = RequestBatcher(engine, max_batch_size=2, max_wait_ms=5000)
    batcher.start()

    ok, bad = await asyncio.wait_for(asyncio.gather(
        batcher.submit("fine", max_tokens=5, temperature=0.7),
        batcher.submit("doomed", max_tokens=5, temperature=0.0),
        return_exceptions=True,
    ), timeout=2)
    await batcher.stop()

    assert ok == "fine|5|0.7"
    assert isinstance(bad, RuntimeError)


@pytest.mark.asyncio
async def test_a_short_engine_reply_fails_the_requests_instead_of_hanging_them():
    engine = RecordingEngine(short=True)
    batcher = RequestBatcher(engine, max_batch_size=2, max_wait_ms=5000)
    batcher.start()

    results = await asyncio.wait_for(asyncio.gather(
        batcher.submit("one"), batcher.submit("two"), return_exceptions=True
    ), timeout=2)
    await batcher.stop()

    assert all(isinstance(r, RuntimeError) and "1 results for 2 prompts" in str(r) for r in results)
