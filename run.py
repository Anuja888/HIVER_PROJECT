#!/usr/bin/env python
"""
Hiver SDE Intern Take-Home — Task Orchestrator

Replaces `make` (not available on Windows). Usage:
    python run.py setup
    python run.py download-data
    python run.py brand-select
    python run.py data-pipeline
    python run.py taxonomy
    python run.py label-mode
    python run.py threshold-sweep
    python run.py eval
    python run.py iaa
    python run.py test
    python run.py run
    python run.py clean
"""
import subprocess
import sys
import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

COMMANDS = {
    "setup": [
        f"{sys.executable} -m pip install -r requirements.txt",
    ],
    "download-data": [
        f"{sys.executable} scripts/download_data.py",
    ],
    "brand-select": [
        f"{sys.executable} scripts/select_brand.py",
    ],
    "data-pipeline": [
        f"{sys.executable} scripts/run_data_pipeline.py",
    ],
    "taxonomy": [
        f"{sys.executable} scripts/build_taxonomy.py",
    ],
    "label-mode": [
        "streamlit run src/ui/app.py",
    ],
    "eval": [
        f"{sys.executable} scripts/run_eval.py",
    ],
    "iaa": [
        f"{sys.executable} scripts/compute_iaa.py",
    ],
    "threshold-sweep": [
        f"{sys.executable} scripts/threshold_sweep.py",
    ],
    "test": [
        f"{sys.executable} -m pytest tests/ -v",
    ],
    "run": [
        "streamlit run src/ui/app.py",
    ],
    "clean": [
        "rmdir /s /q data\\cache 2>nul",
        "rmdir /s /q data\\gold 2>nul",
        "rmdir /s /q reports\\figures 2>nul",
    ],
}


def run_cmd(cmd: str) -> int:
    print(f"\n{'='*60}\n> {cmd}\n{'='*60}")
    return subprocess.call(cmd, shell=True, cwd=REPO_ROOT)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd not in COMMANDS:
        print(f"Unknown command: {cmd}")
        print(f"Available: {', '.join(COMMANDS)}")
        sys.exit(1)
    for c in COMMANDS[cmd]:
        rc = run_cmd(c)
        if rc != 0 and cmd not in ("clean",):
            print(f"Command failed with exit code {rc}")
            sys.exit(rc)


if __name__ == "__main__":
    main()