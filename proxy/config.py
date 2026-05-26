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
class ProxyConfig:
    server: ServerConfig
    models: ModelConfig
    tracking: TrackingConfig

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

        return cls(server=server, models=models, tracking=tracking)

    def resolve_model(self, model: str) -> str:
        return self.models.resolve(model)
