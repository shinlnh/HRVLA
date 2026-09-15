from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/summarize_humanoidarena_baseline_matrix.py"
SPEC = importlib.util.spec_from_file_location("humanoidarena_matrix_summary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def test_wilson_interval_contains_empirical_rate() -> None:
    lower, upper = SUMMARY.wilson_interval(13, 20)
    assert lower < 13 / 20 < upper
    assert SUMMARY.wilson_interval(0, 0) == [0.0, 0.0]


def test_partial_summary_is_explicitly_non_claim(tmp_path: Path) -> None:
    report = SUMMARY.summarize(tmp_path, allow_partial=True)
    assert report["includes_ours"] is False
    assert report["claim_status"] == "partial_non_claim"
    assert len(report["admission_blockers"]) == 2
    assert report["matrix"]["cells_complete"] == 0
    assert report["matrix"]["episodes_complete"] == 0


def test_partial_summary_counts_atomic_episode_evidence(tmp_path: Path) -> None:
    cell = tmp_path / "base_test" / "boxing" / "seed-0" / "episodes"
    cell.mkdir(parents=True)
    for repeat_idx, success in ((0, True), (1, False), (2, True)):
        (cell / f"episode-{repeat_idx}.json").write_text(
            json.dumps(
                {
                    "seed": 0,
                    "repeat_idx": repeat_idx,
                    "success": success,
                    "failure_reason": "success" if success else "timeout",
                    "episode_steps": 10,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    (cell / "interrupted.json").write_text(
        json.dumps(
            {
                "seed": 0,
                "repeat_idx": 3,
                "success": False,
                "failure_reason": "interrupted",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = SUMMARY.summarize(tmp_path, allow_partial=True)
    assert report["claim_status"] == "partial_non_claim"
    assert report["matrix"]["cells_complete"] == 0
    assert report["matrix"]["cells_with_observations"] == 1
    assert report["matrix"]["episodes_complete"] == 3
    assert report["overall"]["successes"] == 2
