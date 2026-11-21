"""Wait for this training process, then evaluate its completed checkpoint."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "ml" / "runs" / "kev4b-injection"
REPORTS = ROOT / "ml" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
STATUS = REPORTS / "completion-status.txt"


def write(message):
    STATUS.write_text(message + "\n", encoding="utf-8")
    print(message, flush=True)


def main():
    pid = int(sys.argv[1])
    started = psutil.Process(pid).create_time()
    write(f"Waiting for training process {pid}")
    while True:
        try:
            proc = psutil.Process(pid)
            active = proc.create_time() == started and proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            active = False
        if not active:
            break
        time.sleep(20)
    if not (RUN / "training_metrics.json").exists() or not (RUN / "head.pt").exists():
        write("Training exited without a complete checkpoint; evaluation was not run")
        return 1
    write("Training complete; evaluating held-out test set")
    with (REPORTS / "evaluation.log").open("w", encoding="utf-8") as log:
        result = subprocess.run([sys.executable, str(ROOT / "ml/scripts/evaluate_guard.py")],
                                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    write("Evaluation complete: ml/reports/kev4b-test.json" if result.returncode == 0
          else f"Evaluation failed with exit code {result.returncode}; see ml/reports/evaluation.log")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
