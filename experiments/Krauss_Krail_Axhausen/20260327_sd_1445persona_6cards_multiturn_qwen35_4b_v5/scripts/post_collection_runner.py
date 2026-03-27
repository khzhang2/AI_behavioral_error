from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_DIR.parents[2]
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"


def resolve_local_torch_dir() -> Path | None:
    for candidate in (
        PROJECT_ROOT / ".python_packages" / "cu118",
        PROJECT_ROOT / ".python_packages" / "cu126",
    ):
        if candidate.exists():
            return candidate
    return None


LOCAL_TORCH_DIR = resolve_local_torch_dir()


def resolve_root_python() -> Path:
    candidates = [
        PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
        PROJECT_ROOT / ".venv" / "bin" / "python",
        Path(sys.executable),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path(sys.executable)


ROOT_PYTHON = resolve_root_python()


def load_progress() -> dict | None:
    progress_path = OUTPUT_DIR / "run_progress.json"
    if not progress_path.exists():
        return None
    return json.loads(progress_path.read_text())


def run_step(script_name: str, extra_env: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    if LOCAL_TORCH_DIR is not None:
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(LOCAL_TORCH_DIR)
            if not existing_pythonpath
            else str(LOCAL_TORCH_DIR) + os.pathsep + existing_pythonpath
        )
    if extra_env:
        env.update(extra_env)
    subprocess.run([str(ROOT_PYTHON), str(EXPERIMENT_DIR / "scripts" / script_name)], check=True, env=env)


def main() -> None:
    log_path = OUTPUT_DIR / "post_collection_runner.log"
    with log_path.open("a", encoding="utf-8") as log_handle:
        while True:
            progress = load_progress()
            if progress is None:
                log_handle.write("waiting: run_progress.json not created yet\n")
                log_handle.flush()
                time.sleep(10)
                continue
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
        run_step("build_comparison.py", {"MPLCONFIGDIR": str(Path(tempfile.gettempdir()) / "matplotlib")})
        log_handle.write("comparison complete, writing summary\n")
        log_handle.flush()
        run_step("summarize_experiment.py")
        log_handle.write("post-processing complete\n")
        log_handle.flush()


if __name__ == "__main__":
    main()
