"""OmegaConf + Hydra-style config loader with CLI override support.

Config files live under ``configs/``. Stage configs use ``defaults:`` lists to
inherit from ``configs/_base/``. This loader resolves those lists without
requiring the full Hydra runtime, so configs can be loaded anywhere (training
scripts, UI server, tests).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

_CONFIGS_DIR = Path(__file__).parents[3] / "configs"


def load_config(
    config_path: str | Path,
    overrides: list[str] | None = None,
) -> DictConfig:
    """Load a YAML config and apply CLI-style dot-list overrides.

    Resolves a ``defaults:`` list by loading each base config from
    ``configs/_base/`` and merging in order, then overlaying the current
    config (``_self_`` semantics).

    Args:
        config_path: Absolute path, or path relative to ``configs/``.
        overrides: List of ``key=value`` strings, e.g.
            ``["training.lr=2e-4", "data.languages=[en,ha]"]``.

    Returns:
        Fully merged and resolved ``DictConfig``.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = _CONFIGS_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    cfg: DictConfig = OmegaConf.load(path)

    if "defaults" in cfg:
        merged: DictConfig = OmegaConf.create({})
        self_cfg = OmegaConf.create({k: v for k, v in cfg.items() if k != "defaults"})
        for entry in cfg.defaults:
            if entry == "_self_":
                merged = OmegaConf.merge(merged, self_cfg)
            else:
                base_path = _CONFIGS_DIR / f"{entry}.yaml"
                if base_path.exists():
                    base = OmegaConf.load(base_path)
                    merged = OmegaConf.merge(merged, base)
        # If _self_ was not listed explicitly, append self at end
        if "_self_" not in cfg.defaults:
            merged = OmegaConf.merge(merged, self_cfg)
        cfg = merged

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    OmegaConf.resolve(cfg)
    return cfg


def to_yaml(cfg: DictConfig) -> str:
    """Serialize a config to a YAML string."""
    return OmegaConf.to_yaml(cfg, resolve=True)


def cfg_to_dict(cfg: DictConfig) -> dict[str, Any]:
    """Convert a config to a plain nested dict."""
    result = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(result, dict)
    return result  # type: ignore[return-value]
