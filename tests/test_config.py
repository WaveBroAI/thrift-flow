"""Tests for ProxyConfig loading — no LiteLLM calls involved."""
import os

import pytest
import yaml

from proxy.config import ProxyConfig


def _write_config(tmp_path, data: dict) -> str:
    config_path = str(tmp_path / "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(data, f)
    return config_path


def test_load_with_valid_yaml(tmp_path):
    """ProxyConfig.load parses a valid config.yaml correctly."""
    config_data = {
        "server": {"host": "127.0.0.1", "port": 9000},
        "models": {
            "aliases": {
                "cheap": "openrouter/minimax/minimax-m2.5",
                "strong": "openai/gpt-4o",
            },
            "default": "cheap",
        },
        "tracking": {"db": "my.db", "enabled": True},
    }
    config_path = _write_config(tmp_path, config_data)

    cfg = ProxyConfig.load(config_path)

    assert cfg.server.host == "127.0.0.1"
    assert cfg.server.port == 9000
    assert cfg.models.aliases["cheap"] == "openrouter/minimax/minimax-m2.5"
    assert cfg.models.aliases["strong"] == "openai/gpt-4o"
    assert cfg.models.default == "cheap"
    assert cfg.tracking.db == "my.db"
    assert cfg.tracking.enabled is True


def test_resolve_model_with_known_alias(tmp_path):
    """resolve_model returns the real model name for a known alias."""
    config_data = {
        "server": {"host": "0.0.0.0", "port": 8888},
        "models": {
            "aliases": {
                "cheap": "openrouter/minimax/minimax-m2.5",
                "medium": "openai/gpt-4o-mini",
            },
            "default": "cheap",
        },
        "tracking": {"db": "tracking.db", "enabled": True},
    }
    config_path = _write_config(tmp_path, config_data)
    cfg = ProxyConfig.load(config_path)

    assert cfg.resolve_model("cheap") == "openrouter/minimax/minimax-m2.5"
    assert cfg.resolve_model("medium") == "openai/gpt-4o-mini"


def test_resolve_model_with_unknown_name_returns_as_is(tmp_path):
    """resolve_model returns the input unchanged if it is not a known alias."""
    config_data = {
        "server": {"host": "0.0.0.0", "port": 8888},
        "models": {
            "aliases": {"cheap": "openrouter/minimax/minimax-m2.5"},
            "default": "cheap",
        },
        "tracking": {"db": "tracking.db", "enabled": True},
    }
    config_path = _write_config(tmp_path, config_data)
    cfg = ProxyConfig.load(config_path)

    assert cfg.resolve_model("openai/gpt-4-turbo") == "openai/gpt-4-turbo"
    assert cfg.resolve_model("some-unknown-model") == "some-unknown-model"


def test_load_uses_defaults_for_missing_fields(tmp_path):
    """ProxyConfig.load applies sensible defaults when optional keys are absent."""
    config_data = {
        "models": {
            "aliases": {"cheap": "openrouter/minimax/minimax-m2.5"},
        },
    }
    config_path = _write_config(tmp_path, config_data)
    cfg = ProxyConfig.load(config_path)

    assert cfg.server.host == "0.0.0.0"
    assert cfg.server.port == 8888
    assert cfg.models.default == "cheap"
    assert cfg.tracking.db == "tracking.db"
    assert cfg.tracking.enabled is True
