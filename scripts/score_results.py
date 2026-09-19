#!/usr/bin/env python3
"""Run the HRVLA scorer from a source checkout without installing it."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hrvla_bench.score import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
