"""BaseTrainer: Accelerate-backed training loop with EventBus integration.

Subclasses implement ``_build_model``, ``_build_dataloaders``, and
``_validation_step`` for each training stage.  The base class owns the
optimiser, scheduler, gradient accumulation, checkpointing, and callback
dispatch.
"""

from __future__ import annotations

import datetime
import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from accelerate import Accelerator
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from orinode.training.callbacks import BestCheckpointCallback, EventEmitterCallback
from orinode.utils.events import EventBus
from orinode.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class TrainingState:
    """Mutable snapshot of the current training state."""

    stage: str = ""
    epoch: int = 0
    global_step: int = 0
    total_steps: int = 0
    train_loss: float = float("nan")
    val_loss: float = float("nan")
    wer: float = float("nan")
    lr: float = 0.0
    grad_norm: float = 0.0
    best_checkpoint: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class BaseTrainer:
    """Accelerate-backed trainer base class.

    Subclasses must implement:
        - ``_build_model(cfg) -> nn.Module``
        - ``_build_dataloaders(cfg) -> tuple[DataLoader, DataLoader]``
        - ``_validation_step(batch, model, state) -> dict[str, float]``
        - ``stage_name`` class attribute

    Args:
        cfg: Full merged OmegaConf config.
        smoke_test: If True, run 2 train steps + 1 val step then exit.
    """

    stage_name: str = "base"

    @staticmethod
    def _generate_unique_run_id(base_name: str) -> str:
        """Generate {base_name}_{YYYYMMDD}_{HHMMSS}_{4hex}."""
        ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        short_hash = uuid.uuid4().hex[:4]
        return f"{base_name}_{ts}_{short_hash}"

    def __init__(self, cfg: DictConfig, smoke_test: bool = False) -> None:
        self.cfg = cfg
        self.smoke_test = smoke_test
        self.state = TrainingState(stage=self.stage_name)

        # Unique run identity — base_run_name from config, run_id unique per launch.
        # Resume case: supply unique_run_id in config to reuse an existing ID.
        self.base_run_name = cfg.get("run_name", self.stage_name)
        self.run_id = cfg.get("unique_run_id") or self._generate_unique_run_id(self.base_run_name)
        print(f"[TRAINER] base_run_name={self.base_run_name}")
        print(f"[TRAINER] run_id={self.run_id}")

        import torch

        mixed_prec = "no" if smoke_test or not torch.cuda.is_available() else "bf16"
        grad_accum = cfg.training.get(
            "gradient_accumulation_steps", cfg.training.get("grad_accum_steps", 1)
        )
        self.accelerator = Accelerator(
            mixed_precision=mixed_prec,
            gradient_accumulation_steps=int(grad_accum),
            log_with=None,
        )

        # EventBus — single shared file per stage, events tagged with unique run_id.
        from orinode.paths import WS

        bus_path = WS.logs / f"{self.stage_name}_events.jsonl"
        self.bus = EventBus(path=bus_path, run_id=self.run_id)

        # Callbacks
        self._callbacks: list[Any] = [EventEmitterCallback(self.bus, cfg=cfg)]

    # ── subclass API ──────────────────────────────────────────────────────────

    def _build_model(self, cfg: DictConfig) -> nn.Module:
        raise NotImplementedError

    def _build_dataloaders(
        self, cfg: DictConfig
    ) -> tuple[DataLoader, DataLoader]:
        raise NotImplementedError

    def _validation_step(
        self,
        batch: Any,
        model: nn.Module,
        state: TrainingState,
    ) -> dict[str, float]:
        raise NotImplementedError

    def _training_step(
        self,
        batch: Any,
        model: nn.Module,
        state: TrainingState,
    ) -> torch.Tensor:
        raise NotImplementedError

    # ── optimiser / scheduler ─────────────────────────────────────────────────

    def _build_optimiser(self, model: nn.Module) -> torch.optim.Optimizer:
        t_cfg = self.cfg.training
        opt_cfg = self.cfg.get("optimizer", {})
        trainable = [p for p in model.parameters() if p.requires_grad]
        lr = float(opt_cfg.get("lr", t_cfg.get("lr", 1e-4)))
        weight_decay = float(opt_cfg.get("weight_decay", t_cfg.get("weight_decay", 0.01)))
        betas = tuple(opt_cfg.get("betas", t_cfg.get("betas", [0.9, 0.999])))
        return torch.optim.AdamW(
            trainable,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
        )

    def _build_scheduler(
        self,
        optimiser: torch.optim.Optimizer,
        num_training_steps: int,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        from torch.optim.lr_scheduler import CosineAnnealingLR

        sched_cfg = self.cfg.get("scheduler", {})
        warmup = int(sched_cfg.get("warmup_steps", self.cfg.training.get("warmup_steps", 500)))
        # Linear warmup + cosine decay via sequential scheduler
        from torch.optim.lr_scheduler import LinearLR, SequentialLR

        warmup_sched = LinearLR(optimiser, start_factor=1e-3, total_iters=warmup)
        cosine_sched = CosineAnnealingLR(
            optimiser, T_max=max(1, num_training_steps - warmup)
        )
        return SequentialLR(
            optimiser,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup],
        )

    # ── checkpoint helpers ────────────────────────────────────────────────────

    def _save_checkpoint(self, model: nn.Module, path: Path) -> None:
        if not self.accelerator.is_main_process:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        self.accelerator.save(
            {
                "model": self.accelerator.unwrap_model(model).state_dict(),
                "global_step": self.state.global_step,
                "val_wer": self.state.wer,
                "val_loss": self.state.val_loss,
            },
            path,
        )
        log.info(f"Checkpoint saved → {path}")
        for cb in self._callbacks:
            if hasattr(cb, "on_checkpoint_saved"):
                cb.on_checkpoint_saved(self.state, path)

    # ── callback dispatch ─────────────────────────────────────────────────────

    def _call(self, hook: str, **kwargs: Any) -> None:
        for cb in self._callbacks:
            fn = getattr(cb, hook, None)
            if fn is not None:
                fn(self.state, **kwargs)

    # ── smoke-test path (no Accelerate) ──────────────────────────────────────

    def _smoke_train(
        self,
        model: nn.Module,
        optimiser: torch.optim.Optimizer,
        scheduler: Any,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> TrainingState:
        """Minimal 2-step training loop used by --smoke-test; no Accelerate."""
        from orinode.paths import WS

        ckpt_dir = WS.models_checkpoints / self.run_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._callbacks.append(
            BestCheckpointCallback(
                save_dir=ckpt_dir,
                metric="val_wer",
                mode="min",
                max_keep=2,
                save_fn=lambda p: self._save_checkpoint(model, p),
            )
        )
        self._call("on_train_start")
        max_grad_norm = self.cfg.training.get(
            "grad_clip", self.cfg.training.get("max_grad_norm", 1.0)
        )

        try:
            self.state.epoch = 0
            model.train()
            for step, batch in enumerate(train_loader):
                if step >= 2:
                    break
                optimiser.zero_grad()
                loss = self._training_step(batch, model, self.state)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimiser.step()
                scheduler.step()
                self.state.global_step += 1
                self.state.train_loss = loss.item()
                self.state.lr = scheduler.get_last_lr()[0]
                self.state.grad_norm = gn.item()
                self._call("on_step_end")

            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    metrics = self._validation_step(batch, model, self.state)
                    self.state.val_loss = metrics.get("val_loss", float("nan"))
                    break
            self._call("on_validation_end")
            self._call("on_epoch_end")

        except Exception as exc:
            self._call("on_error", exc=exc)
            raise
        finally:
            self._call("on_train_end")

        return self.state

    # ── main train loop ───────────────────────────────────────────────────────

    def train(self) -> TrainingState:
        """Run the full training loop.

        Respects the following config keys under ``training:``:
            max_steps           — stop after N optimizer steps (0 = run full epochs)
            eval_interval       — run validation every N optimizer steps (0 = epoch-end only)
            checkpoint_interval — save checkpoint every N optimizer steps
            gradient_accumulation_steps (or grad_accum_steps)
            grad_clip (or max_grad_norm)

        Returns:
            Final ``TrainingState``.
        """
        cfg = self.cfg
        t_cfg = cfg.training

        model = self._build_model(cfg)
        train_loader, val_loader = self._build_dataloaders(cfg)

        if self.smoke_test:
            # Skip Accelerate wrapping — plain PyTorch avoids contiguity issues
            # on CPU/MPS that arise from Accelerate's model patching.
            optimiser = self._build_optimiser(model)
            scheduler = self._build_scheduler(optimiser, 10)
            return self._smoke_train(model, optimiser, scheduler, train_loader, val_loader)

        # Resolve config keys (support old and new names)
        max_steps = int(t_cfg.get("max_steps", 0))        # 0 = unlimited
        eval_every = int(t_cfg.get("eval_interval", 0))   # 0 = epoch-end only
        grad_clip = float(t_cfg.get("grad_clip", t_cfg.get("max_grad_norm", 1.0)))
        grad_accum = int(
            t_cfg.get("gradient_accumulation_steps", t_cfg.get("grad_accum_steps", 1))
        )

        steps_per_epoch = len(train_loader)
        optimizer_steps_per_epoch = max(1, steps_per_epoch // grad_accum)

        if max_steps > 0:
            total_steps = max_steps
            epochs = math.ceil(max_steps / optimizer_steps_per_epoch)
        else:
            epochs = int(t_cfg.get("num_epochs", 3))
            total_steps = epochs * optimizer_steps_per_epoch

        self.state.total_steps = total_steps

        optimiser = self._build_optimiser(model)
        scheduler = self._build_scheduler(optimiser, total_steps)

        # Resume from checkpoint if specified
        resume_from = cfg.get("training", {}).get("resume_from_checkpoint", None)
        if resume_from:
            resume_path = Path(resume_from)
            if resume_path.is_dir():
                resume_path = resume_path / "best.pt"
            if resume_path.exists():
                ckpt = torch.load(resume_path, map_location="cpu", weights_only=True)
                raw_model = model.module if hasattr(model, "module") else model
                raw_model.load_state_dict(ckpt["model"], strict=False)
                self.state.global_step = ckpt.get("global_step", 0)
                log.info(
                    f"Resumed from {resume_path}  global_step={self.state.global_step}"
                )
            else:
                log.warning(f"Resume checkpoint not found: {resume_path}")

        # Accelerate prepares model, optimiser, loaders, scheduler
        model, optimiser, train_loader, val_loader, scheduler = (
            self.accelerator.prepare(model, optimiser, train_loader, val_loader, scheduler)
        )

        from orinode.paths import WS

        ckpt_dir = WS.models_checkpoints / self.run_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._callbacks.append(
            BestCheckpointCallback(
                save_dir=ckpt_dir,
                metric="val_wer",
                mode="min",
                max_keep=2,
                save_fn=lambda p: self._save_checkpoint(model, p),
            )
        )

        self._call("on_train_start")
        done = False

        try:
            for epoch in range(epochs):
                if done:
                    break
                self.state.epoch = epoch
                model.train()

                for batch in train_loader:
                    if done:
                        break

                    is_opt_step = False
                    with self.accelerator.accumulate(model):
                        loss = self._training_step(batch, model, self.state)
                        self.accelerator.backward(loss)

                        if self.accelerator.sync_gradients:
                            is_opt_step = True
                            gn = self.accelerator.clip_grad_norm_(
                                model.parameters(), grad_clip
                            )
                            self.state.grad_norm = (
                                gn.item() if hasattr(gn, "item") else float(gn)
                            )
                        optimiser.step()
                        scheduler.step()
                        optimiser.zero_grad()

                    if not is_opt_step:
                        continue

                    self.state.global_step += 1
                    self.state.train_loss = loss.item()
                    self.state.lr = scheduler.get_last_lr()[0]
                    self._call("on_step_end")

                    # Step-level eval — BestCheckpointCallback saves step_{N}.pt
                    if eval_every > 0 and self.state.global_step % eval_every == 0:
                        model.eval()
                        val_metrics = self._run_validation(val_loader, model)
                        self.state.val_loss = val_metrics.get("val_loss", math.nan)
                        self.state.wer = val_metrics.get("wer", math.nan)
                        self._call("on_validation_end")
                        model.train()

                    if max_steps > 0 and self.state.global_step >= max_steps:
                        done = True

                # End-of-epoch validation when not using step-level eval
                if not done and eval_every == 0:
                    model.eval()
                    val_metrics = self._run_validation(val_loader, model)
                    self.state.val_loss = val_metrics.get("val_loss", math.nan)
                    self.state.wer = val_metrics.get("wer", math.nan)
                    self._call("on_validation_end")

                self._call("on_epoch_end")

        except Exception as exc:
            self._call("on_error", exc=exc)
            raise
        finally:
            self._call("on_train_end")
            self.accelerator.end_training()

        return self.state

    def _run_validation(
        self,
        loader: DataLoader,
        model: nn.Module,
    ) -> dict[str, float]:
        """Iterate over validation loader and aggregate metrics."""
        total_loss = 0.0
        n_batches = 0
        max_val_batches = 1 if self.smoke_test else int(1e9)

        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= max_val_batches:
                    break
                metrics = self._validation_step(batch, model, self.state)
                total_loss += metrics.get("val_loss", 0.0)
                n_batches += 1

        return {"val_loss": total_loss / max(n_batches, 1)}
