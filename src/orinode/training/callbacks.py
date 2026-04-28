"""Training callbacks: checkpoint saving and event emission."""

from __future__ import annotations

import contextlib
import json
import math
import shutil
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from orinode.training.trainer import TrainingState
    from orinode.utils.events import EventBus


class BestCheckpointCallback:
    """Save and manage checkpoints on each validation end.

    Policy:
        - Saves ``step_{N}.pt`` on every eval.
        - Tracks best by ``metric`` (default: ``val_wer``, mode=min).
        - Strictly-less-than replacement: ties do NOT displace the current best.
        - Keeps ``max_keep`` most-recent step files plus the best step file on disk.
        - Deletes oldest non-best step files beyond that limit.
        - Writes ``best.pt`` (atomic shutil.copy) and ``best_metadata.json``.

    Args:
        save_dir: Directory for checkpoints.
        metric: ``TrainingState`` attribute to track (``"val_wer"``).
        mode: ``"min"`` (lower is better) or ``"max"``.
        max_keep: Number of most-recent step files to keep besides best.
        save_fn: Callable ``(path: Path) -> None`` that writes the checkpoint file.
    """

    def __init__(
        self,
        save_dir: Path,
        metric: str = "val_wer",
        mode: str = "min",
        max_keep: int = 2,
        save_fn: Callable[[Path], None] | None = None,
    ) -> None:
        self.save_dir = save_dir
        self.metric = metric
        self.mode = mode
        self.max_keep = max_keep
        self.save_fn = save_fn
        self._best: float | None = None
        self._best_step: int | None = None

    def _is_better(self, value: float) -> bool:
        """Return True only if value is strictly better than current best."""
        if self._best is None:
            return True
        return value < self._best if self.mode == "min" else value > self._best

    def on_validation_end(self, state: TrainingState) -> None:
        from orinode.utils.logging import get_logger

        _log = get_logger(__name__)

        step = state.global_step
        wer: float = getattr(state, "wer", math.nan)
        val_loss: float = getattr(state, "val_loss", math.nan)

        if self.metric == "val_wer":
            metric_val = wer
        elif self.metric == "val_loss":
            metric_val = val_loss
        else:
            metric_val = float(getattr(state, self.metric, math.nan))

        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Save this eval's step checkpoint
        step_path = self.save_dir / f"step_{step}.pt"
        if self.save_fn is not None:
            self.save_fn(step_path)
        wer_str = f"{wer:.4f}" if not math.isnan(wer) else "N/A"
        loss_str = f"{val_loss:.4f}" if not math.isnan(val_loss) else "N/A"
        _log.info(f"[Checkpoint] Saved: {step_path.name}  WER={wer_str}  loss={loss_str}")

        # Update best only if strictly better and metric is not NaN
        if not math.isnan(metric_val) and self._is_better(metric_val):
            self._best = metric_val
            self._best_step = step

            # Atomic copy to best.pt
            best_path = self.save_dir / "best.pt"
            tmp_best = self.save_dir / "best.pt.tmp"
            shutil.copy2(step_path, tmp_best)
            tmp_best.rename(best_path)

            # Atomic write of best_metadata.json
            meta: dict[str, Any] = {
                "best_step": step,
                "best_wer": None if math.isnan(wer) else round(wer, 6),
                "best_loss": None if math.isnan(val_loss) else round(val_loss, 6),
                "updated_at": datetime.utcnow().isoformat(),
            }
            tmp_meta = self.save_dir / "best_metadata.json.tmp"
            tmp_meta.write_text(json.dumps(meta, indent=2))
            tmp_meta.rename(self.save_dir / "best_metadata.json")
            _log.info(
                f"[Checkpoint] Best updated → step={step}  WER={wer_str}  loss={loss_str}"
            )
        else:
            if self._best_step is not None:
                _log.info(
                    f"[Checkpoint] Best unchanged: step={self._best_step}  "
                    f"{self.metric}={self._best:.4f}"
                )
            else:
                _log.info("[Checkpoint] Metric is NaN — best not updated (no WER yet)")

        state.best_checkpoint = self.save_dir / "best.pt"

        # Rolling policy: keep max_keep most-recent + best step file
        all_step_files = sorted(
            self.save_dir.glob("step_*.pt"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        keep: set[Path] = set()
        for p in all_step_files[-self.max_keep :]:
            keep.add(p)
        if self._best_step is not None:
            keep.add(self.save_dir / f"step_{self._best_step}.pt")

        deleted: list[str] = []
        for p in all_step_files:
            if p not in keep and p.exists():
                p.unlink()
                deleted.append(p.name)

        on_disk = sorted(
            [p.name for p in keep if p.exists()],
            key=lambda n: int(n.split("_")[1].split(".")[0]),
        )
        _log.info(f"[Checkpoint] On disk: {on_disk}  Deleted: {deleted or ['none']}")


class EventEmitterCallback:
    """Emit structured events to the EventBus at each training milestone.

    Args:
        bus: ``EventBus`` instance to write events to.
        cfg: OmegaConf config — used to populate ``train_start`` fields.
    """

    def __init__(self, bus: EventBus, cfg: DictConfig | None = None) -> None:
        self.bus = bus
        self._cfg = cfg
        # Bug 2 fix: set adaptively in on_train_start based on total_steps
        self._emit_every: int = 1

    def on_train_start(self, state: TrainingState) -> None:
        # Honor logging_interval from config; fall back to adaptive total//100
        log_interval = None
        if self._cfg is not None:
            log_interval = self._cfg.training.get("logging_interval", None)
        if log_interval is not None:
            self._emit_every = max(1, int(log_interval))
        else:
            total = state.total_steps
            self._emit_every = max(1, total // 100) if total >= 10 else 1

        # Bug 1: populate stage / run_name / config_yaml from cfg
        stage = 0
        run_name = ""
        config_yaml = ""
        augmentation: str | None = None

        cfg = self._cfg
        if cfg is not None:
            from omegaconf import OmegaConf

            with contextlib.suppress(Exception):
                stage = int(cfg.get("stage", 0))
            run_name = self.bus.run_id  # unique run_id, not stale config value
            try:
                config_yaml = OmegaConf.to_yaml(cfg, resolve=True)
            except Exception:  # noqa: BLE001
                config_yaml = str(cfg)

        from orinode.utils.events import TrainStartEvent

        self.bus.emit(
            TrainStartEvent(
                stage=stage,
                run_name=run_name,
                config_yaml=config_yaml,
                total_steps=state.total_steps,
                augmentation=augmentation,
            )
        )

    def on_step_end(self, state: TrainingState) -> None:
        # Bug 2: emit every N steps adaptively; always emit step 1 (first step)
        if state.global_step % self._emit_every != 0 and state.global_step != 1:
            return
        from orinode.utils.events import StepEvent

        self.bus.emit(
            StepEvent(
                step=state.global_step,
                loss=state.train_loss,
                lr=state.lr,
                grad_norm=state.grad_norm,
                epoch=state.epoch,
            )
        )

    def on_validation_end(self, state: TrainingState) -> None:
        from orinode.utils.events import EvalEvent

        wer_dict: dict[str, float] = (
            {"overall": state.wer} if not math.isnan(state.wer) else {}
        )
        eval_loss = state.val_loss if not math.isnan(state.val_loss) else None
        self.bus.emit(
            EvalEvent(
                step=state.global_step,
                wer=wer_dict,
                eval_loss=eval_loss,
            )
        )

    def on_epoch_end(self, state: TrainingState) -> None:
        from orinode.utils.events import EpochCompleteEvent

        self.bus.emit(
            EpochCompleteEvent(
                epoch=state.epoch,
                step=state.global_step,
                avg_loss=state.train_loss,
            )
        )

    def on_checkpoint_saved(self, state: TrainingState, path: Path) -> None:
        from orinode.utils.events import CheckpointSavedEvent

        self.bus.emit(
            CheckpointSavedEvent(
                step=state.global_step,
                path=str(path),
            )
        )

    def on_train_end(self, state: TrainingState) -> None:
        from orinode.utils.events import TrainEndEvent

        self.bus.emit(
            TrainEndEvent(
                total_steps=state.global_step,
                best_eval_loss=(
                    state.val_loss if not math.isnan(state.val_loss) else None
                ),
            )
        )

    def on_error(self, state: TrainingState, exc: BaseException) -> None:
        from orinode.utils.events import ErrorEvent

        self.bus.emit(ErrorEvent(message=str(exc)))

    def _unused_any(self) -> Any:  # keeps TYPE_CHECKING import used
        return None
