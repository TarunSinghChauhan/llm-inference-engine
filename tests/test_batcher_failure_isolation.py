"""One bad prompt must not fail its batch-mates, and stopping must not strand requests."""
import asyncio
import threading

import pytest

from engine.serving.batcher import RequestBatcher


class FakeEngine:
    def __init__(self):
        self.calls = []

    def generate(self, prompts, max_tokens, temperature):
        self.calls.append(list(prompts))
        if any("BAD" in p for p in prompts):
            raise ValueError("Requested tokens exceed context window")
        return [f"ok:{p}" for p in prompts]


async def submit_all(batcher, prompts):
    return await asyncio.wait_for(
        asyncio.gather(*[batcher.submit(p) for p in prompts], return_exceptions=True), timeout=5
    )


# ---- failure isolation ------------------------------------------------------------

@pytest.mark.asyncio
async def test_one_bad_prompt_fails_only_its_own_request():
    engine = FakeEngine()
    b = RequestBatcher(engine, max_batch_size=8, max_wait_ms=30)
    b.start()
    try:
        out = await submit_all(b, ["good one", "BAD prompt", "good two"])
    finally:
        await b.stop()
    assert out[0] == "ok:good one" and out[2] == "ok:good two"
    assert isinstance(out[1], ValueError)


@pytest.mark.asyncio
async def test_the_batch_is_tried_whole_first_then_one_by_one():
    engine = FakeEngine()
    b = RequestBatcher(engine, max_batch_size=8, max_wait_ms=30)
    b.start()
    try:
        await submit_all(b, ["a", "BAD", "c"])
    finally:
        await b.stop()
    assert engine.calls[0] == ["a", "BAD", "c"]
    assert engine.calls[1:] == [["a"], ["BAD"], ["c"]]


@pytest.mark.asyncio
async def test_a_failing_single_request_is_not_retried():
    engine = FakeEngine()
    b = RequestBatcher(engine, max_batch_size=8, max_wait_ms=10)
    b.start()
    try:
        out = await submit_all(b, ["BAD"])
    finally:
        await b.stop()
    assert isinstance(out[0], ValueError)
    assert engine.calls == [["BAD"]]


@pytest.mark.asyncio
async def test_a_healthy_batch_is_one_call_as_before():
    engine = FakeEngine()
    b = RequestBatcher(engine, max_batch_size=8, max_wait_ms=30)
    b.start()
    try:
        out = await submit_all(b, ["a", "b", "c"])
    finally:
        await b.stop()
    assert out == ["ok:a", "ok:b", "ok:c"]
    assert engine.calls == [["a", "b", "c"]]


@pytest.mark.asyncio
async def test_when_the_engine_is_down_everyone_gets_an_error_and_nobody_hangs():
    class Down:
        def generate(self, **kw):
            raise RuntimeError("engine down")

    b = RequestBatcher(Down(), max_batch_size=8, max_wait_ms=20)
    b.start()
    try:
        out = await submit_all(b, ["a", "b", "c"])
    finally:
        await b.stop()
    assert all(isinstance(o, RuntimeError) and "engine down" in str(o) for o in out)


@pytest.mark.asyncio
async def test_a_result_count_mismatch_still_fails_the_requests():
    class Short:
        def generate(self, prompts, **kw):
            return prompts[:-1]

    b = RequestBatcher(Short(), max_batch_size=8, max_wait_ms=20)
    b.start()
    try:
        out = await submit_all(b, ["a", "b"])
    finally:
        await b.stop()
    assert all(isinstance(o, RuntimeError) for o in out)


# ---- stopping ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stop_fails_requests_that_were_still_queued():
    b = RequestBatcher(FakeEngine(), max_batch_size=100, max_wait_ms=60_000)
    b.start()
    tasks = [asyncio.create_task(b.submit(f"p{i}")) for i in range(3)]
    await asyncio.sleep(0.05)
    await b.stop()
    out = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5)
    assert all(isinstance(o, RuntimeError) and "stopped" in str(o) for o in out)


@pytest.mark.asyncio
async def test_stop_fails_a_batch_that_is_being_generated():
    started, release = threading.Event(), threading.Event()

    class Slow:
        def generate(self, prompts, **kw):
            started.set()
            release.wait(5)
            return ["late" for _ in prompts]

    b = RequestBatcher(Slow(), max_batch_size=8, max_wait_ms=10)
    b.start()
    task = asyncio.create_task(b.submit("p"))
    try:
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(0.02)
        assert started.is_set()
        await b.stop()
        with pytest.raises(RuntimeError, match="stopped"):
            await asyncio.wait_for(task, timeout=5)
    finally:
        release.set()


@pytest.mark.asyncio
async def test_submit_after_stop_is_refused_instead_of_hanging():
    b = RequestBatcher(FakeEngine(), max_batch_size=8, max_wait_ms=10)
    b.start()
    await b.stop()
    with pytest.raises(RuntimeError, match="stopped"):
        await asyncio.wait_for(b.submit("late"), timeout=2)


@pytest.mark.asyncio
async def test_stop_is_safe_to_call_twice_or_without_start():
    b = RequestBatcher(FakeEngine())
    await b.stop()
    b.start()
    await b.stop()
    await b.stop()


@pytest.mark.asyncio
async def test_a_batcher_can_be_started_again_after_a_stop():
    b = RequestBatcher(FakeEngine(), max_batch_size=8, max_wait_ms=10)
    b.start()
    await b.stop()
    b.start()
    try:
        assert await asyncio.wait_for(b.submit("again"), timeout=5) == "ok:again"
    finally:
        await b.stop()
