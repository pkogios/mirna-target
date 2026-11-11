"""Model registry: names used on the command line and in configs/models.yaml."""
from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_FILE = Path(__file__).resolve().parents[2] / "configs" / "models.yaml"


def load_config(path=CONFIG_FILE) -> dict:
    return yaml.safe_load(Path(path).read_text())


def build_model(name: str, seed: int = 0, rounds: int | None = None, config: dict | None = None, **overrides):
    """Create an unfitted model from its entry in the config file."""
    config = config if config is not None else load_config()
    if name not in config:
        raise KeyError(f"Unknown model '{name}'. Available: {', '.join(config)}")
    settings = {**config[name], **overrides}
    kind = settings.pop("kind")
    if kind == "neural":
        from .neural import NeuralModel
        return NeuralModel(seed=seed, rounds=rounds, **settings)
    from . import classical
    return getattr(classical, kind)(seed=seed, rounds=rounds, **settings)
