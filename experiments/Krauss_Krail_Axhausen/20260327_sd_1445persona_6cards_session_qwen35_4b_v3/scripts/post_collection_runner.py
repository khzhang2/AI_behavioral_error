from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_DIR.parents[2]
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
ROOT_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"


def load_progress() -> dict:
    return json.loads((OUTPUT_DIR / "run_progress.json").read_text())


def run_step(script_name: str, extra_env: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    subprocess.run([str(ROOT_PYTHON), str(EXPERIMENT_DIR / "scripts" / script_name)], check=True, env=env)


def main() -> None:
    log_path = OUTPUT_DIR / "post_collection_runner.log"
    with log_path.open("a", encoding="utf-8") as log_handle:
        while True:
            progress = load_progress()
            completed = int(progress["completed_respondents"])
            target = int(progress["target_respondents"])
            log_handle.write(f"waiting: completed={completed} target={target}\n")
            log_handle.flush()
            if completed >= target:
                break
            time.sleep(60)

        log_handle.write("collection complete, starting estimation\n")
        log_handle.flush()
        run_step("estimate_mixed_choice_model.py", {"OMP_NUM_THREADS": "1"})
        log_handle.write("estimation complete, building comparison\n")
        log_handle.flush()
        run_step("build_comparison.py", {"MPLCONFIGDIR": "/tmp/matplotlib"})
        log_handle.write("comparison complete, writing summary\n")
        log_handle.flush()
        run_step("summarize_experiment.py")
        log_handle.write("post-processing complete\n")
        log_handle.flush()


if __name__ == "__main__":
    main()
