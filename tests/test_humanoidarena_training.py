from __future__ import annotations

import json
from pathlib import Path

from hrvla_bench.humanoidarena_training import (
    build_training_command,
    build_validation_command,
    build_hidden_command,
    evaluation_complete,
    load_training_lock,
    select_global_checkpoint,
    validate_selection,
)


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "config/humanoidarena-gr00t-training.lock.json"


def test_locked_training_and_validation_commands_are_split_safe() -> None:
    lock = load_training_lock(LOCK)
    training = build_training_command(ROOT, Path("python"), lock, 2)
    assert training[training.index("--dataset-path") + 1].endswith("/train")
    assert training[training.index("--dataloader-num-workers") + 1] == "4"
    assert training[training.index("--seed") + 1] == "2"
    validation = build_validation_command(ROOT, Path("python"), lock, 1, 200)
    assert validation[validation.index("--dataset-path") + 1].endswith("/validation")
    assert "/heldout" not in " ".join(validation)
    trajectory_start = validation.index("--trajectory-ids") + 1
    condition_start = validation.index("--conditions")
    assert validation[trajectory_start:condition_start] == [str(value) for value in range(70)]
    hidden = build_hidden_command(ROOT, Path("python"), lock, 1, 200)
    assert hidden[hidden.index("--dataset-path") + 1].endswith("/heldout")
    assert "/validation" not in hidden[hidden.index("--dataset-path") + 1]


def test_global_selection_uses_macro_task_mse_and_lower_step_tie(tmp_path: Path) -> None:
    lock = load_training_lock(LOCK)
    lock["dataset"]["path"] = "view"
    metadata = tmp_path / "view/validation/meta/episodes.jsonl"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        "".join(
            json.dumps(
                {
                    "episode_index": episode,
                    "source_dataset": f"task-{episode // 10}/sonic_refpose_v3_1",
                }
            )
            + "\n"
            for episode in range(70)
        )
    )
    evaluation = tmp_path / "evaluation"
    for seed in (0, 1, 2):
        for step, mse in ((100, 0.3), (200, 0.2), (300, 0.2)):
            path = evaluation / f"seed-{seed}/checkpoint-{step}/metrics.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "rows": [
                            {"trajectory_id": episode, "condition": "clean", "mse": mse}
                            for episode in range(70)
                        ]
                    }
                )
            )
    report = select_global_checkpoint(tmp_path, lock, evaluation)
    assert report["selected_step"] == 200
    assert report["hidden_test_accessed"] is False
    assert len(report["selection_sha256"]) == 64
    assert validate_selection(report, lock) == 200


def test_evaluation_completion_requires_the_exact_cross_product(tmp_path: Path) -> None:
    path = tmp_path / "evaluation"
    path.mkdir()
    rows = [
        {"trajectory_id": trajectory, "condition": condition}
        for trajectory in range(2)
        for condition in ("clean", "noise")
    ]
    (path / "metrics.json").write_text(json.dumps({"rows": rows}))
    assert evaluation_complete(path, 2, ["clean", "noise"])
    rows.append(dict(rows[-1]))
    (path / "metrics.json").write_text(json.dumps({"rows": rows}))
    assert not evaluation_complete(path, 2, ["clean", "noise"])
