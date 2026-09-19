#!/usr/bin/env python3
"""Materialize the complete tracked HumanoidArena report and visual bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    commands = (
        [sys.executable, "scripts/summarize_humanoidarena_baseline_matrix.py"],
        [sys.executable, "scripts/render_benchmark_evidence.py"],
        [sys.executable, "scripts/render_humanoidarena_internal_results.py"],
        [sys.executable, "scripts/render_humanoidarena_video_evidence.py"],
    )
    for command in commands:
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
