import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.modules.setdefault("llama_cpp", MagicMock())

import pytest
from fastapi.testclient import TestClient

import api.main as main_module
from api.queue import RequestQueue


@pytest.fixture
def client():
    with patch.object(main_module.engine, "load", MagicMock()), \
         patch.object(main_module.queue, "connect", AsyncMock()), \
         patch.object(main_module.batcher, "start", MagicMock()), \
         patch.object(main_module.batcher, "stop", AsyncMock()):
        with TestClient(main_module.app) as c:
            yield c


def _patch_queue(depth=0, req_id="req-x", latency=1.0):
    return (
        patch.object(main_module.queue, "current_depth", AsyncMock(return_value=depth)),
        patch.object(main_module.queue, "track_start", AsyncMock(return_value=req_id)),
        patch.object(main_module.queue, "track_end", AsyncMock(return_value=latency)),
        patch.object(main_module.queue, "track_abort", AsyncMock()),
    )


def test_a_failed_generation_releases_its_queue_slot(client):
    p1, p2, p3, p4 = _patch_queue(req_id="req-fail")
    with p1, p2, p3 as track_end, p4 as track_abort, \
         patch.object(main_module.batcher, "submit", AsyncMock(side_effect=RuntimeError("engine down"))):
        with pytest.raises(RuntimeError, match="engine down"):
            client.post("/generate", json={"prompts": ["hi"]})
    track_abort.assert_awaited_once_with("req-fail")
    track_end.assert_not_awaited()


def test_a_successful_generation_does_not_abort(client):
    p1, p2, p3, p4 = _patch_queue()
    with p1, p2, p3 as track_end, p4 as track_abort, \
         patch.object(main_module.batcher, "submit", AsyncMock(return_value="ok")):
        resp = client.post("/generate", json={"prompts": ["hi"]})
    assert resp.status_code == 200
    track_end.assert_awaited_once()
    track_abort.assert_not_awaited()


def test_a_failing_abort_does_not_hide_the_original_error(client):
    p1, p2, p3, p4 = _patch_queue()
    with p1, p2, p3, patch.object(main_module.queue, "track_abort", AsyncMock(side_effect=ConnectionError("redis gone"))), \
         patch.object(main_module.batcher, "submit", AsyncMock(side_effect=RuntimeError("engine down"))):
        with pytest.raises(RuntimeError, match="engine down"):
            client.post("/generate", json={"prompts": ["hi"]})


@pytest.mark.parametrize("body", [
    {"prompts": []},
    {"prompts": ["x"] * 33},
    {"prompts": ["x"], "max_tokens": 0},
    {"prompts": ["x"], "max_tokens": 10**6},
    {"prompts": ["x"], "temperature": -0.1},
    {"prompts": ["x"], "temperature": 5},
])
def test_invalid_generate_requests_are_rejected_before_any_work(client, body):
    p1, p2, p3, p4 = _patch_queue()
    with p1, p2 as track_start, p3, p4, patch.object(main_module.batcher, "submit", AsyncMock()) as submit:
        resp = client.post("/generate", json=body)
    assert resp.status_code == 422
    track_start.assert_not_awaited()
    submit.assert_not_awaited()


def test_the_documented_bounds_are_accepted(client):
    p1, p2, p3, p4 = _patch_queue()
    with p1, p2, p3, p4, patch.object(main_module.batcher, "submit", AsyncMock(return_value="ok")):
        resp = client.post("/generate", json={"prompts": ["x"] * 32, "max_tokens": 4096, "temperature": 2.0})
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_track_abort_gives_back_the_depth_slot_and_deletes_the_request_hash():
    q = RequestQueue()
    q._client = AsyncMock()
    await q.track_abort("abc")
    q._client.decr.assert_awaited_once_with("inference:queue_depth")
    q._client.delete.assert_awaited_once_with("inference:req:abc")
    q._client.lpush.assert_not_awaited()  # an aborted request is not a latency sample
