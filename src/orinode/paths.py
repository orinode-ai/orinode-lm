"""
paths.py — single source of truth for all workspace/ paths.

Every module that needs a file path imports from here. Nothing hard-codes
paths directly. The workspace root is controlled by the ``ORINODE_WORKSPACE``
environment variable (default ``./workspace`` relative to cwd).
"""

from __future__ import annotations

import os
from pathlib import Path


def _workspace_root() -> Path:
    raw = os.environ.get("ORINODE_WORKSPACE", "./workspace")
    return Path(raw).expanduser().resolve()


class WorkspacePaths:
    """Typed path bundle for the full workspace/ layout.

    Use the module-level ``WS`` singleton everywhere; do not construct
    additional instances unless you need an isolated workspace (e.g. tests
    override ``ORINODE_WORKSPACE`` before importing).
    """

    def __init__(self, root: Path) -> None:
        self.root = root

        # ── data ──────────────────────────────────────────────────────────────
        self.data = root / "data"
        self.data_raw = self.data / "raw"
        self.data_processed = self.data / "processed"
        self.data_manifests = self.data / "manifests"

        self.raw_naijavoices = self.data_raw / "naijavoices"
        self.raw_afrispeech = self.data_raw / "afrispeech_200"
        self.raw_bibletts = self.data_raw / "bibletts"
        self.raw_common_voice = self.data_raw / "common_voice"
        self.raw_crowdsourced_cs = self.data_raw / "crowdsourced_cs"

        # ── models ────────────────────────────────────────────────────────────
        self.models = root / "models"
        self.models_base = self.models / "base"
        self.models_checkpoints = self.models / "checkpoints"

        # ── logs ──────────────────────────────────────────────────────────────
        self.logs = root / "logs"
        self.logs_wandb = self.logs / "wandb"
        self.logs_training = self.logs / "training"

        # ── evals ─────────────────────────────────────────────────────────────
        self.evals = root / "evals"

        # ── cache ─────────────────────────────────────────────────────────────
        self.cache = root / "cache"
        self.cache_huggingface = self.cache / "huggingface"
        self.cache_transformers = self.cache / "transformers"

    # ── per-run helpers ───────────────────────────────────────────────────────

    def checkpoint_dir(self, run_id: str) -> Path:
        """Return the checkpoint directory for ``run_id``."""
        return self.models_checkpoints / run_id

    def training_log_dir(self, run_id: str) -> Path:
        """Return the log directory for ``run_id``."""
        return self.logs_training / run_id

    def events_file(self, run_id: str) -> Path:
        """Return the JSONL event bus file path for ``run_id``."""
        return self.training_log_dir(run_id) / "events.jsonl"

    def stdout_log(self, run_id: str) -> Path:
        return self.training_log_dir(run_id) / "stdout.log"

    def eval_dir(self, run_id: str) -> Path:
        return self.evals / run_id

    def processed_corpus_dir(self, corpus: str) -> Path:
        return self.data_processed / corpus

    # ── setup ─────────────────────────────────────────────────────────────────

    def ensure_all(self) -> None:
        """Create every workspace subdirectory (idempotent)."""
        dirs = [
            self.data_raw,
            self.raw_naijavoices,
            self.raw_afrispeech,
            self.raw_bibletts,
            self.raw_common_voice,
            self.raw_crowdsourced_cs,
            self.data_processed,
            self.data_manifests,
            self.models_base,
            self.models_checkpoints,
            self.logs_wandb,
            self.logs_training,
            self.evals,
            self.cache_huggingface,
            self.cache_transformers,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return f"WorkspacePaths(root={self.root})"


# Module-level singleton. Import this; do not re-instantiate.
WS: WorkspacePaths = WorkspacePaths(_workspace_root())


def ensure_workspace() -> WorkspacePaths:
    """Ensure all workspace dirs exist and return the ``WS`` singleton.

    Call at process entry points (UI server, training scripts) for belt-and-
    braces directory creation. Safe to call multiple times.
    """
    WS.ensure_all()
    return WS
