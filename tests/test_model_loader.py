import os
import sys
from unittest.mock import patch, MagicMock

sys.modules["llama_cpp"] = MagicMock()

import pytest

from engine.serving.model_loader import EngineConfig, InferenceEngine


def test_engine_config_can_be_overridden_directly():
    # EngineConfig's defaults read env vars at class-definition time (module
    # import), not per-instantiation, so monkeypatching env vars after import
    # has no effect. Direct field overrides are the correct way to customize
    # a config instance at runtime.
    config = EngineConfig(model_path="/custom/path.gguf", n_gpu_layers=10, n_ctx=2048, n_batch=128, n_threads=4)
    assert config.model_path == "/custom/path.gguf"
    assert config.n_gpu_layers == 10
    assert config.n_ctx == 2048
    assert config.n_batch == 128
    assert config.n_threads == 4


def test_engine_config_defaults_when_no_env_vars(monkeypatch):
    for var in ["MODEL_PATH", "N_GPU_LAYERS", "MAX_MODEL_LEN", "N_BATCH", "N_THREADS"]:
        monkeypatch.delenv(var, raising=False)

    config = EngineConfig()
    assert config.model_path == "./models/qwen2.5-3b-instruct-q4_k_m.gguf"
    assert config.n_gpu_layers == 20
    assert config.n_ctx == 4096
    assert config.n_batch == 256


def test_llm_property_raises_before_load():
    engine = InferenceEngine(config=EngineConfig())
    with pytest.raises(RuntimeError, match="Engine not loaded"):
        _ = engine.llm


def test_llm_property_returns_loaded_instance():
    engine = InferenceEngine(config=EngineConfig())
    fake_llm = MagicMock()
    engine._llm = fake_llm
    assert engine.llm is fake_llm


def test_generate_loops_over_prompts_and_extracts_text():
    engine = InferenceEngine(config=EngineConfig())
    fake_llm = MagicMock()
    fake_llm.side_effect = [
        {"choices": [{"text": "response one"}]},
        {"choices": [{"text": "response two"}]},
    ]
    engine._llm = fake_llm

    results = engine.generate(prompts=["prompt one", "prompt two"], max_tokens=100, temperature=0.5)

    assert results == ["response one", "response two"]
    assert fake_llm.call_count == 2


def test_load_constructs_llama_with_config_values():
    config = EngineConfig(model_path="/fake.gguf", n_gpu_layers=5, n_ctx=1024, n_batch=64, n_threads=2)
    engine = InferenceEngine(config=config)

    with patch("engine.serving.model_loader.Llama") as mock_llama_cls:
        mock_llama_cls.return_value = MagicMock()
        result = engine.load()

        mock_llama_cls.assert_called_once_with(
            model_path="/fake.gguf",
            n_gpu_layers=5,
            n_ctx=1024,
            n_batch=64,
            n_threads=2,
            verbose=False,
        )
        assert result is engine
