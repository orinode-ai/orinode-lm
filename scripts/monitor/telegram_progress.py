"""Telegram progress monitor for Stage 1 training.

This is an example monitor specific to Orinode's Stage 1 encoder
adaptation training run. Copy and adapt the constants
(MAX_STEPS, EVENTBUS_PATH) for other stages.

Reads the most recent metrics from the EventBus JSONL, the checkpoint
directory (best_metadata.json), and nvidia-smi. Formats a progress
summary and sends to Telegram every N minutes.

Requires environment variables (in .env):
    TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID

Designed to be robust: failures in Telegram or log parsing must
NEVER affect the training process. This script is a fully separate
background process.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

# ── Configuration ────────────────────────────────────────────────────────────

POLL_INTERVAL_SEC = 30 * 60  # 30 minutes
TRAINING_PID_PATH = Path("/tmp/stage1_train.pid")
EVENTBUS_PATH = Path("workspace/logs/stage1_encoder_events.jsonl")
CHECKPOINT_BASE = Path("workspace/models/checkpoints")
HEARTBEAT_PATH = Path("/tmp/telegram_progress.heartbeat")
MAX_STEPS = 20_000  # Stage 1 full run

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


# ── Telegram ──────────────────────────────────────────────────────────────────

def send_telegram(message: str) -> bool:
    """Send a message to Telegram. Returns True on success."""
    if not BOT_TOKEN or not CHAT_ID:
        print("[ERROR] Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
        return False

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = urlencode(
        {
            "chat_id": CHAT_ID,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": "true",
        }
    ).encode("utf-8")

    try:
        req = Request(url, data=data, method="POST")
        with urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                return True
            print(f"[WARN] Telegram returned status {resp.status}")
            return False
    except (URLError, OSError) as e:
        print(f"[WARN] Telegram send failed: {e}")
        return False


# ── Process check ─────────────────────────────────────────────────────────────

def is_training_alive() -> bool:
    """Check if the training process is still running via PID file."""
    if not TRAINING_PID_PATH.exists():
        return False
    try:
        pid = int(TRAINING_PID_PATH.read_text().strip())
        os.kill(pid, 0)  # signal 0 = existence check only
        return True
    except (ValueError, ProcessLookupError, PermissionError):
        return False


# ── Metrics parsing ───────────────────────────────────────────────────────────

def parse_latest_metrics() -> dict:
    """Read the most recent training metrics from EventBus JSONL + checkpoint dir.

    Returns dict with any of: step, train_loss, val_loss, wer, lr, grad_norm,
    best_step, best_wer, best_loss, best_updated_at, ckpt_steps,
    gpu_mem_gb, gpu_util_pct, gpu_temp_c.
    """
    metrics: dict = {}

    if EVENTBUS_PATH.exists():
        try:
            with open(EVENTBUS_PATH) as f:
                lines = f.readlines()

            # Scan last 500 lines; find most recent step, eval, and train_start events
            recent = lines[-500:] if len(lines) > 500 else lines
            last_step: dict | None = None
            last_eval: dict | None = None
            current_run_id: str | None = None

            for line in recent:
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = ev.get("type", "")
                if t == "train_start":
                    current_run_id = ev.get("run_id")
                elif t == "step":
                    last_step = ev
                    if current_run_id is None:
                        current_run_id = ev.get("run_id")
                elif t == "eval":
                    last_eval = ev

            if last_step:
                metrics["step"] = last_step.get("step")
                metrics["train_loss"] = last_step.get("loss")
                metrics["lr"] = last_step.get("lr")
                metrics["grad_norm"] = last_step.get("grad_norm")

            if last_eval:
                metrics["val_loss"] = last_eval.get("eval_loss")
                raw_wer = last_eval.get("wer")
                if isinstance(raw_wer, dict):
                    metrics["wer"] = raw_wer.get("overall")
                elif isinstance(raw_wer, (float, int)):
                    metrics["wer"] = float(raw_wer)

            # Read best_metadata.json and list step checkpoints on disk
            if current_run_id:
                ckpt_dir = CHECKPOINT_BASE / current_run_id
                meta_path = ckpt_dir / "best_metadata.json"
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text())
                        metrics["best_step"] = meta.get("best_step")
                        metrics["best_wer"] = meta.get("best_wer")
                        metrics["best_loss"] = meta.get("best_loss")
                        metrics["best_updated_at"] = meta.get("updated_at")
                    except Exception:
                        pass
                try:
                    step_files = sorted(
                        ckpt_dir.glob("step_*.pt"),
                        key=lambda p: int(p.stem.split("_")[1]),
                    )
                    metrics["ckpt_steps"] = [int(p.stem.split("_")[1]) for p in step_files]
                except Exception:
                    pass

        except Exception as e:
            print(f"[WARN] EventBus parse failed: {e}")

    # GPU stats via nvidia-smi
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            if len(parts) >= 1:
                metrics["gpu_mem_gb"] = int(parts[0]) / 1024
            if len(parts) >= 2:
                metrics["gpu_util_pct"] = int(parts[1])
            if len(parts) >= 3:
                metrics["gpu_temp_c"] = int(parts[2])
    except Exception:
        pass

    return metrics


# ── Formatting ────────────────────────────────────────────────────────────────

def estimate_eta(step: int | None, start_time: datetime) -> str:
    if not step:
        return "calculating..."
    elapsed = (datetime.utcnow() - start_time).total_seconds()
    if elapsed < 120 or step < 10:
        return "calculating..."
    rate = step / elapsed  # steps/sec
    remaining_sec = (MAX_STEPS - step) / rate
    eta_dt = datetime.utcnow() + timedelta(seconds=remaining_sec)
    hours = remaining_sec / 3600
    return f"{hours:.1f}h  eta {eta_dt.strftime('%H:%M')} UTC"


def format_message(metrics: dict, start_time: datetime, cycle: int) -> str:
    step = metrics.get("step")
    train_loss = metrics.get("train_loss")
    val_loss = metrics.get("val_loss")
    wer = metrics.get("wer")
    lr = metrics.get("lr")
    grad_norm = metrics.get("grad_norm")
    gpu_mem = metrics.get("gpu_mem_gb")
    gpu_util = metrics.get("gpu_util_pct")
    gpu_temp = metrics.get("gpu_temp_c")
    best_step = metrics.get("best_step")
    best_wer = metrics.get("best_wer")
    best_loss = metrics.get("best_loss")
    best_at = metrics.get("best_updated_at")
    ckpt_steps: list[int] = metrics.get("ckpt_steps", [])

    elapsed_h = (datetime.utcnow() - start_time).total_seconds() / 3600
    pct = f"{100 * step / MAX_STEPS:.1f}%" if step else "?"
    eta = estimate_eta(step, start_time)

    lines: list[str] = [
        f"*Stage 1 — update #{cycle}*",
        f"`step {step}/{MAX_STEPS}` ({pct})  elapsed: {elapsed_h:.1f}h",
        f"ETA: {eta}",
        "",
    ]

    # Latest eval block
    has_eval = val_loss is not None or wer is not None
    if has_eval:
        lines.append("*Latest eval:*")
        if val_loss is not None:
            lines.append(f"  loss: `{val_loss:.4f}`")
        if wer is not None:
            lines.append(f"  WER:  `{wer * 100:.2f}%`")
        lines.append("")

    # Best checkpoint block
    if best_step is not None:
        lines.append("*Best checkpoint:*")
        wer_s = f"{best_wer * 100:.2f}%" if best_wer is not None else "N/A"
        loss_s = f"{best_loss:.4f}" if best_loss is not None else "N/A"
        lines.append(f"  step {best_step} — WER {wer_s}, loss {loss_s}")
        if best_at:
            try:
                at_dt = datetime.fromisoformat(best_at)
                mins_ago = int((datetime.utcnow() - at_dt).total_seconds() / 60)
                lines.append(f"  set {mins_ago} min ago")
            except Exception:
                pass
        lines.append("")
    else:
        lines.append("Best: not yet set (awaiting first eval)")
        lines.append("")

    # Checkpoints on disk
    if ckpt_steps:
        parts = []
        for s in sorted(ckpt_steps, reverse=True):
            tag = " ★" if s == best_step else ""
            parts.append(f"{s}{tag}")
        lines.append(f"On disk: {', '.join(parts)}")

    # Train loss + lr
    if train_loss is not None:
        lr_s = f"  lr: `{lr:.2e}`" if lr is not None else ""
        gn_s = f"  gn: `{grad_norm:.3f}`" if grad_norm is not None else ""
        lines.append(f"train loss: `{train_loss:.4f}`{lr_s}{gn_s}")

    # GPU
    gpu_parts = []
    if gpu_util is not None:
        gpu_parts.append(f"{gpu_util}%")
    if gpu_mem is not None:
        gpu_parts.append(f"{gpu_mem:.1f}GB")
    if gpu_temp is not None:
        gpu_parts.append(f"{gpu_temp}°C")
    if gpu_parts:
        lines.append(f"GPU: `{'  '.join(gpu_parts)}`")

    return "\n".join(lines)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    if not BOT_TOKEN or not CHAT_ID:
        print("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
        sys.exit(1)

    print(f"Telegram monitor starting at {datetime.utcnow().isoformat()} UTC")
    print(f"Updates every {POLL_INTERVAL_SEC // 60} min  |  EventBus: {EVENTBUS_PATH}")

    # Wait up to 5 min for training PID file to appear
    for _ in range(10):
        if is_training_alive():
            break
        print("Waiting for training process...")
        time.sleep(30)
    else:
        send_telegram("Orinode monitor: training process not detected after 5 min. Exiting.")
        sys.exit(1)

    start_time = datetime.utcnow()
    send_telegram(
        f"*Stage 1 — started*\n"
        f"Monitor active, updates every 30 min.\n"
        f"Started: `{start_time.strftime('%H:%M UTC')}`"
    )

    cycle = 0
    last_send_ts = time.monotonic()

    try:
        while is_training_alive():
            HEARTBEAT_PATH.write_text(datetime.utcnow().isoformat())
            time.sleep(60)  # wake every minute to check alive; batch into 30-min sends

            if time.monotonic() - last_send_ts < POLL_INTERVAL_SEC:
                continue

            cycle += 1
            try:
                metrics = parse_latest_metrics()
                msg = format_message(metrics, start_time, cycle)
                sent = send_telegram(msg)
                print(f"[{datetime.utcnow().isoformat()}] Update #{cycle} sent={sent}")
            except Exception as e:
                print(f"[ERROR] Update #{cycle}: {e}\n{traceback.format_exc()}")
            finally:
                last_send_ts = time.monotonic()

    except KeyboardInterrupt:
        send_telegram("Orinode monitor: stopped manually.")
        return

    except Exception as e:
        send_telegram(f"Orinode monitor: crashed — `{e}`\nTraining continues unaffected.")
        raise

    # Training finished
    try:
        metrics = parse_latest_metrics()
        step = metrics.get("step", "?")
        loss = metrics.get("train_loss")
        wer = metrics.get("wer")
        best_step = metrics.get("best_step")
        best_wer = metrics.get("best_wer")
        elapsed_h = (datetime.utcnow() - start_time).total_seconds() / 3600

        parts = [
            "*Stage 1 — complete*",
            f"Final step: `{step}`",
        ]
        if loss is not None:
            parts.append(f"Final train loss: `{loss:.4f}`")
        if wer is not None:
            parts.append(f"Final WER: `{wer * 100:.2f}%`")
        if best_step is not None:
            wer_s = f"{best_wer * 100:.2f}%" if best_wer is not None else "N/A"
            parts.append(f"Best checkpoint: step {best_step}, WER {wer_s}")
        parts.append(f"Total elapsed: `{elapsed_h:.1f}h`")
        send_telegram("\n".join(parts))
    except Exception:
        send_telegram("*Stage 1 — complete*\nCould not parse final metrics.")


if __name__ == "__main__":
    main()
