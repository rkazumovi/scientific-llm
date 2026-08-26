"""
scientific-llm - Step 4b: training callbacks (logging + checkpointing).

trainer.py uses a hand-written PyTorch loop rather than HuggingFace's
Trainer/TRL's SFTTrainer (see trainer.py's docstring for why), so this is
not a TrainerCallback in the framework sense - it is a small,
dependency-free logger the loop calls into at each step. Two jobs:
  1. Print + append a CSV row per step (loss components, learning rate,
     GPU memory) so a run can be inspected/plotted later without needing
     a tracking service account (W&B, etc. - none of which this project
     assumes you have set up).
  2. Track the best validation loss seen so far, so trainer.py knows
     when to save a "best" checkpoint versus just the "latest" one.

Run directly:
    python src\\training\\callbacks.py
(exercises the logger against a few fake steps - no model or GPU needed.)
"""

import csv
import sys
import time
from pathlib import Path

import torch


class TrainingLogger:
    def __init__(self, log_dir: str, run_name: str = "run"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / f"{run_name}.csv"
        self.best_val_loss = float("inf")
        self._start_time = time.time()

        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["step", "elapsed_s", "ce_loss", "physics_loss", "total_loss", "lr", "gpu_mem_gb"]
            )

    def log_step(
        self,
        step: int,
        ce_loss: float,
        physics_loss: float | None,
        total_loss: float,
        lr: float,
    ) -> None:
        gpu_mem_gb = 0.0
        if torch.cuda.is_available():
            gpu_mem_gb = torch.cuda.memory_allocated() / (1024**3)

        elapsed = time.time() - self._start_time
        physics_str = f"{physics_loss:.4f}" if physics_loss is not None else "-"

        print(
            f"  step {step:5d} | ce={ce_loss:.4f} | physics={physics_str} | "
            f"total={total_loss:.4f} | lr={lr:.2e} | gpu={gpu_mem_gb:.2f}GB | "
            f"elapsed={elapsed:.1f}s"
        )

        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [step, f"{elapsed:.1f}", ce_loss, physics_loss if physics_loss is not None else "",
                 total_loss, lr, f"{gpu_mem_gb:.3f}"]
            )

    def is_best(self, val_loss: float) -> bool:
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            return True
        return False


def main() -> int:
    import random

    logger = TrainingLogger(log_dir="outputs/logs", run_name="callbacks_demo")
    print(f"Writing demo log to {logger.csv_path}")

    for step in range(1, 6):
        ce = 2.0 - step * 0.1 + random.uniform(-0.02, 0.02)
        physics = 0.5 - step * 0.05 if step % 2 == 0 else None
        total = ce + (physics or 0.0)
        logger.log_step(step, ce_loss=ce, physics_loss=physics, total_loss=total, lr=2e-4)

    is_best = logger.is_best(1.0)
    still_best = logger.is_best(1.5)

    if not logger.csv_path.exists():
        print("FAIL: CSV log file was not created.")
        return 1
    if not is_best or still_best:
        print("FAIL: is_best() tracking logic is wrong.")
        return 1

    print(f"\nPASS: logged 5 steps, CSV written to {logger.csv_path}, best-loss tracking correct.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
