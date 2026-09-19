#!/usr/bin/env python3
"""Run HRVLA benchmark tools from a source checkout."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hrvla_bench.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
