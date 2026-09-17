from __future__ import annotations

import copy
import json
from pathlib import Path

from hrvla_bench.humanoidarena_rt_training import (
    build_training_command,
    build_validation_command,
    load_rt_lock,
    select_checkpoint,
)
from hrvla_bench.plan import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
RT_LOCK = ROOT / "config/humanoidarena-rt-training.lock.json"
COMMON_LOCK = json.loads(
    (ROOT / "config/humanoidarena-gr00t-training.lock.json").read_text(encoding="utf-8")
)


def test_rt_commands_initialize_from_seed_matched_common_and_never_validation_data() -> None:
    lock = load_rt_lock(RT_LOCK)
    command = build_training_command(
        ROOT, Path("python"), lock, COMMON_LOCK, 200, "gr00t_str_rt", 2
    )
    assert "seed-2/checkpoints/checkpoint-200" in command[command.index("--base-model-path") + 1]
    assert command[command.index("--dataset-path") + 1].endswith("humanoidarena-str-rt/train")
    assert command[command.index("--dataloader-num-workers") + 1] == "4"
    validation = build_validation_command(
        ROOT, Path("python"), lock, "gr00t_st_rt", 1, 300
    )
    assert validation[validation.index("--dataset-path") + 1].endswith(
        "humanoidarena-st-rt/validation"
    )
    assert validation[validation.index("--throughput-batch-size") + 1] == "30"
    assert validation[validation.index("--prefetch-workers") + 1] == "16"
    assert validation[validation.index("--prefetch-pending-per-worker") + 1] == "1"
    assert "/heldout" not in " ".join(validation)


def test_rt_selection_is_macro_task_balanced_and_tie_breaks_to_lower_step(tmp_path: Path) -> None:
    lock = load_rt_lock(RT_LOCK)
    lock = copy.deepcopy(lock)
    lock["output_root"] = "runs"
    lock["subtask_rt_dataset"]["path"] = "st-data"
    metadata = tmp_path / "st-data/validation/meta/episodes.jsonl"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        "".join(
            json.dumps(
                {
                    "episode_index": index,
                    "source_dataset": f"task-{index // 10}/source",
                }
            )
            + "\n"
            for index in range(70)
        ),
        encoding="utf-8",
    )
    manifest_core = {"schema_version": 1, "split": "validation", "episodes": 70}
    manifest = {**manifest_core, "manifest_sha256": canonical_sha256(manifest_core)}
    manifest_path = tmp_path / "st-data/manifests/validation.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    lock["subtask_rt_dataset"]["manifests"]["validation"] = manifest["manifest_sha256"]
    for seed in (0, 1, 2):
        for step, mse in ((100, 0.3), (200, 0.2), (300, 0.2)):
            metrics = (
                tmp_path
                / f"runs/st-rt/seed-{seed}/validation/checkpoint-{step}/metrics.json"
            )
            metrics.parent.mkdir(parents=True)
            metrics.write_text(
                json.dumps(
                    {
                        "rows": [
                            {"trajectory_id": index, "condition": "clean", "mse": mse}
                            for index in range(70)
                        ]
                    }
                ),
                encoding="utf-8",
            )
    report = select_checkpoint(tmp_path, lock, "gr00t_st_rt")
    assert report["selected_step"] == 200
    assert report["hidden_test_accessed"] is False
    assert len(report["selection_sha256"]) == 64
