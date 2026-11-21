"""/health must work with the real config, and the Redis URL must come from the environment."""
import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.modules["llama_cpp"] = MagicMock()

import pytest
from fastapi.testclient import TestClient

import api.main as main_module
from api.queue import DEFAULT_REDIS_URL, RequestQueue
from engine.serving.model_loader import EngineConfig


# ---- model_name --------------------------------------------------------------

@pytest.mark.parametrize("path,expected", [
    ("./models/qwen2.5-3b-instruct-q4_k_m.gguf", "qwen2.5-3b-instruct-q4_k_m.gguf"),
    ("/app/models/llama.gguf", "llama.gguf"),
    ("C:\\models\\phi.gguf", "phi.gguf"),
    ("plain.gguf", "plain.gguf"),
])
def test_model_name_is_the_file_name(path, expected):
    assert EngineConfig(model_path=path).model_name == expected


# ---- /health with the real EngineConfig (not a mock that happens to have the attribute) ----

@pytest.fixture
def real_config_client():
    with patch.object(main_module.engine, "load", MagicMock()), \
         patch.object(main_module.queue, "connect", AsyncMock()), \
         patch.object(main_module.batcher, "start", MagicMock()), \
         patch.object(main_module.batcher, "stop", AsyncMock()), \
         patch.object(main_module.engine, "config", EngineConfig(model_path="/app/models/tiny.gguf")):
        with TestClient(main_module.app) as c:
            yield c


def test_health_works_with_the_real_engine_config(real_config_client):
    resp = real_config_client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model": "tiny.gguf"}


# ---- Redis URL -----------------------------------------------------------------

def test_default_is_localhost_when_nothing_is_set(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    assert RequestQueue()._redis_url == DEFAULT_REDIS_URL == "redis://localhost:6379/0"


def test_environment_variable_overrides_the_default(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://redis:6379/0")
    assert RequestQueue()._redis_url == "redis://redis:6379/0"


def test_an_explicit_argument_beats_the_environment(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://redis:6379/0")
    assert RequestQueue("redis://other:1/2")._redis_url == "redis://other:1/2"


@pytest.mark.asyncio
async def test_connect_uses_the_configured_url(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://redis:6379/0")
    with patch("api.queue.redis.from_url") as from_url:
        await RequestQueue().connect()
    from_url.assert_called_once_with("redis://redis:6379/0", decode_responses=True)


def test_compose_points_the_app_at_the_redis_service():
    from pathlib import Path
    compose = (Path(__file__).parent.parent / "docker-compose.yml").read_text()
    assert "REDIS_URL=redis://redis:6379/0" in compose
