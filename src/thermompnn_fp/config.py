from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

from .types import LocalPaths, ModelConfig, ProjectConfig, TrainConfig

T = TypeVar("T")


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _coerce_dataclass(dataclass_type: type[T], payload: dict[str, Any] | None) -> T:
    payload = payload or {}
    allowed = {field.name for field in fields(dataclass_type)}
    return dataclass_type(**{key: value for key, value in payload.items() if key in allowed})


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at {path}, got {type(data)!r}")
    return data


def load_project_config(
    config_path: str | Path,
    local_paths_path: str | Path | None = None,
) -> ProjectConfig:
    config_payload = load_yaml(config_path)
    local_payload = load_yaml(local_paths_path) if local_paths_path else {}
    merged = _merge_dicts(config_payload, {"local_paths": local_payload})
    return ProjectConfig(
        name=merged.get("name", "thermompnn-fp"),
        model=_coerce_dataclass(ModelConfig, merged.get("model")),
        training=_coerce_dataclass(TrainConfig, merged.get("training")),
        local_paths=_coerce_dataclass(LocalPaths, merged.get("local_paths")),
    )


def dump_project_config(config: ProjectConfig, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(config), handle, sort_keys=False)


def dataclass_to_dict(instance: Any) -> dict[str, Any]:
    if not is_dataclass(instance):
        raise TypeError("Expected a dataclass instance.")
    return asdict(instance)
