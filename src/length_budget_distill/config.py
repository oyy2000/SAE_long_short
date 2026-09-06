"""Configuration loading and path resolution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


Config = Dict[str, Any]


def load_config(path: str) -> Config:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["_config_path"] = str(config_path)
    return config


def resolve_path(
    path_value: Optional[str], config: Optional[Config] = None
) -> Optional[Path]:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    if config and config.get("_config_path"):
        candidate = Path(config["_config_path"]).resolve().parent / path
        if candidate.exists():
            return candidate
    return cwd_path
