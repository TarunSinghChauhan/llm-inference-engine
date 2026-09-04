import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.modules["llama_cpp"] = MagicMock()

import pytest
from fastapi.testclient import TestClient

import api.main as main_module


@pytest.fixture
def client():
    with patch.object(main_module.engine, "load", MagicMock()), \
         patch.object(main_module.queue, "connect", AsyncMock()), \
         patch.object(main_module.batcher, "start", MagicMock()), \
         patch.object(main_module.batcher, "stop", AsyncMock()), \
         patch.object(main_module.engine, "config", MagicMock(model_name="test-model")):
        with TestClient(main_module.app) as c:
            yield c


def test_generate_returns_completions_and_metadata(client):
    with patch.object(main_module.queue, "current_depth", AsyncMock(return_value=3)), \
         patch.object(main_module.queue, "track_start", AsyncMock(return_value="req-1")), \
         patch.object(main_module.queue, "track_end", AsyncMock(return_value=42.5)), \
         patch.object(main_module.batcher, "submit", AsyncMock(side_effect=["out1", "out2"])):
        resp = client.post("/generate", json={"prompts": ["hi", "there"]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["completions"] == ["out1", "out2"]
    assert body["latency_ms"] == 42.5
    assert body["queue_depth_at_request"] == 3


def test_generate_submits_each_prompt_with_request_params(client):
    with patch.object(main_module.queue, "current_depth", AsyncMock(return_value=0)), \
         patch.object(main_module.queue, "track_start", AsyncMock(return_value="req-2")), \
         patch.object(main_module.queue, "track_end", AsyncMock(return_value=10.0)), \
         patch.object(main_module.batcher, "submit", AsyncMock(return_value="ok")) as mock_submit:
        client.post("/generate", json={"prompts": ["only one"], "max_tokens": 64, "temperature": 0.2})
    mock_submit.assert_awaited_once_with("only one", max_tokens=64, temperature=0.2)


def test_generate_uses_default_max_tokens_and_temperature(client):
    with patch.object(main_module.queue, "current_depth", AsyncMock(return_value=0)), \
         patch.object(main_module.queue, "track_start", AsyncMock(return_value="req-3")), \
         patch.object(main_module.queue, "track_end", AsyncMock(return_value=5.0)), \
         patch.object(main_module.batcher, "submit", AsyncMock(return_value="ok")) as mock_submit:
        client.post("/generate", json={"prompts": ["p"]})
    mock_submit.assert_awaited_once_with("p", max_tokens=256, temperature=0.7)


def test_metrics_returns_queue_depth_and_latencies(client):
    with patch.object(main_module.queue, "recent_latencies", AsyncMock(return_value=[1.0, 2.0, 3.0])), \
         patch.object(main_module.queue, "current_depth", AsyncMock(return_value=5)):
        resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["queue_depth"] == 5
    assert body["sample_size"] == 3
    assert body["recent_latencies_ms"] == [1.0, 2.0, 3.0]


def test_health_returns_ok_status_and_model_name(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model": "test-model"}
