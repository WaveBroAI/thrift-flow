from __future__ import annotations

import yaml
from dataclasses import dataclass, field


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8888


@dataclass
class ModelConfig:
    aliases: dict[str, str]
    default: str = "cheap"

    def resolve(self, model: str) -> str:
        """Return aliases[model] if it exists, else return model as-is."""
        return self.aliases.get(model, model)


@dataclass
class TrackingConfig:
    db: str = "tracking.db"
    enabled: bool = True


@dataclass
class RoutingConfig:
    """Phase 2 adaptive routing config. Disabled by default.

    Requires GROQ_API_KEY (or another provider key) if categorizer_model
    points to a provider-specific model (e.g. groq/llama-3.1-8b-instant).
    Has no effect until model: "auto" wiring is added in server.py (PR B2).
    """

    enabled: bool = False
    categorizer_model: str | None = None          # e.g. "groq/llama-3.1-8b-instant"
    categorizer_api_base: str | None = None       # custom endpoint override
    categorizer_api_key_env: str | None = None    # env var holding the API key
    db: str = "tracking.db"                       # shared with TrackingConfig (separate table)
    tier_mapping_version: str = "v1"
    confidence_threshold: float = 0.7             # min confidence for pool eligibility
    min_prompt_length_for_pool: int = 10          # min prompt chars for pool eligibility
    session_ttl_seconds: int = 1800               # session context cache TTL (seconds)
    categorizer_timeout: float = 5.0              # LLM categorizer timeout (seconds)
    max_session_cache_size: int = 1000            # max in-process session cache entries
    # Embedding router (Phase 2)
    embedding_enabled: bool | str = False         # False / "shadow" / True
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_k: int = 5
    embedding_min_pool_size: int = 20
    embedding_pool_cache_ttl: float = 300.0       # seconds


@dataclass
class ProxyConfig:
    server: ServerConfig
    models: ModelConfig
    tracking: TrackingConfig
    routing: RoutingConfig = field(default_factory=RoutingConfig)

    @classmethod
    def load(cls, path: str = "config.yaml") -> "ProxyConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}

        server_raw = raw.get("server", {})
        server = ServerConfig(
            host=server_raw.get("host", "0.0.0.0"),
            port=server_raw.get("port", 8888),
        )

        models_raw = raw.get("models", {})
        models = ModelConfig(
            aliases=models_raw.get("aliases", {}),
            default=models_raw.get("default", "cheap"),
        )

        tracking_raw = raw.get("tracking", {})
        tracking = TrackingConfig(
            db=tracking_raw.get("db", "tracking.db"),
            enabled=tracking_raw.get("enabled", True),
        )

        routing_raw = raw.get("routing", {})
        routing = RoutingConfig(
            enabled=routing_raw.get("enabled", False),
            categorizer_model=routing_raw.get("categorizer_model") or None,
            categorizer_api_base=routing_raw.get("categorizer_api_base") or None,
            categorizer_api_key_env=routing_raw.get("categorizer_api_key_env") or None,
            db=routing_raw.get("db", tracking_raw.get("db", "tracking.db")),
            tier_mapping_version=routing_raw.get("tier_mapping_version", "v1"),
            confidence_threshold=routing_raw.get("confidence_threshold", 0.7),
            min_prompt_length_for_pool=routing_raw.get("min_prompt_length_for_pool", 10),
            session_ttl_seconds=routing_raw.get("session_ttl_seconds", 1800),
            categorizer_timeout=routing_raw.get("categorizer_timeout", 5.0),
            max_session_cache_size=routing_raw.get("max_session_cache_size", 1000),
            embedding_enabled=routing_raw.get("embedding_enabled", False),
            embedding_model=routing_raw.get("embedding_model", "intfloat/multilingual-e5-small"),
            embedding_k=int(routing_raw.get("embedding_k", 5)),
            embedding_min_pool_size=int(routing_raw.get("embedding_min_pool_size", 20)),
            embedding_pool_cache_ttl=float(routing_raw.get("embedding_pool_cache_ttl", 300.0)),
        )

        return cls(server=server, models=models, tracking=tracking, routing=routing)

    def resolve_model(self, model: str) -> str:
        return self.models.resolve(model)
