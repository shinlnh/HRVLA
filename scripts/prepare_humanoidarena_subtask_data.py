#!/usr/bin/env python3
"""Materialize audited ST labels over the frozen HumanoidArena data splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrvla_bench.plan import canonical_sha256, load_json  # noqa: E402
from hrvla_bench.subtask_relabel import segment_subtasks  # noqa: E402


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _program_index(programs: dict[str, Any]) -> tuple[dict[str, list[int]], list[dict[str, Any]]]:
    indices: dict[str, list[int]] = {}
    tasks = []
    for program in programs["tasks"]:
        task_key = program["humanoidarena_task_key"]
        indices[task_key] = []
        for skill in program["skills"]:
            task_index = len(tasks)
            indices[task_key].append(task_index)
            tasks.append(
                {
                    "task_index": task_index,
                    "task": skill["instruction"],
                    "task_id": program["task_id"],
                    "task_key": task_key,
                    "skill_id": skill["id"],
                }
            )
    return indices, tasks


def _source_task_key(row: dict[str, Any], programs: dict[str, Any]) -> str:
    source = str(row["source_dataset"]).split("/", 1)[0]
    for program in programs["tasks"]:
        task_key = program["humanoidarena_task_key"]
        if FAST_TASK_DATASETS[task_key] == source:
            return task_key
    raise ValueError(f"source dataset is not in the frozen method programs: {source}")


FAST_TASK_DATASETS = {
    "doubledesk": "HOI_double_desk",
    "football": "HOI_football",
    "pp_box": "HOI_pp_box",
    "boxing": "HSI_boxing",
    "open_door": "HSI_open_door",
    "sit_sofa": "HSI_sit_sofa",
    "vision_navi": "HSI_vision_navi",
}


def _adjudication_index(lock: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    policy = lock["subtask_rt_dataset"].get("temporal_video_adjudication")
    if not policy or policy.get("status") != "ready":
        return {}
    path = (ROOT / policy["report_path"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"frozen temporal adjudication report is missing: {path}")
    report = load_json(path)
    audit_sha256 = report.get("audit_sha256")
    core = {key: value for key, value in report.items() if key != "audit_sha256"}
    if canonical_sha256(core) != audit_sha256 or audit_sha256 != policy["report_sha256"]:
        raise ValueError("temporal adjudication report differs from the frozen lock")
    if report.get("status") != "pass":
        raise ValueError("temporal adjudication report was not admitted")
    if report.get("source_manifest_sha256") != lock["source_dataset"]["manifest_sha256"]:
        raise ValueError("temporal adjudication used a different source dataset")
    if report.get("method_program_sha256") != lock["method_program_sha256"]:
        raise ValueError("temporal adjudication used different method programs")
    if report.get("model_revision") != policy["model_revision"]:
        raise ValueError("temporal adjudication used a different model revision")
    output: dict[tuple[str, int], dict[str, Any]] = {}
    for row in report["records"]:
        if not row["consensus"].get("accepted"):
            continue
        key = (str(row["split"]), int(row["episode_index"]))
        if key in output:
            raise ValueError(f"duplicate temporal adjudication record: {key}")
        output[key] = row
    return output


def _episode_arrays(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    import pandas as pd

    frame = pd.read_parquet(path)
    states = np.stack(frame["observation.state"].map(lambda value: np.asarray(value, dtype=np.float32)))
    actions = np.stack(frame["action"].map(lambda value: np.asarray(value, dtype=np.float32)))
    return frame, states, actions


def _analyze_split(
    source: Path,
    split: str,
    programs: dict[str, Any],
    minimum_phase_frames: int,
    adjudications: dict[tuple[str, int], dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_split = source / split
    episode_rows = [
        json.loads(line)
        for line in (source_split / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    audits = []
    for row in episode_rows:
        episode = int(row["episode_index"])
        parquet = source_split / f"data/chunk-000/episode_{episode:06d}.parquet"
        _frame, states, actions = _episode_arrays(parquet)
        task_key = _source_task_key(row, programs)
        program = next(
            item for item in programs["tasks"] if item["humanoidarena_task_key"] == task_key
        )
        adjudication = (adjudications or {}).get((split, episode))
        segmentation = segment_subtasks(
            task_key,
            states,
            actions,
            skill_count=len(program["skills"]),
            minimum_phase_frames=minimum_phase_frames,
            boundary_override=(
                list(adjudication["consensus"]["boundaries"])
                if adjudication is not None
                else None
            ),
            boundary_override_audit=(adjudication["consensus"] if adjudication else None),
        )
        audits.append(
            {
                "episode_index": episode,
                "source_dataset": row["source_dataset"],
                **segmentation.audit,
            }
        )
    return episode_rows, audits


def materialize_split(
    source: Path,
    output_root: Path,
    split: str,
    programs: dict[str, Any],
    lock: dict[str, Any],
    adjudications: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    import pandas as pd

    if split not in {"train", "validation"}:
        raise ValueError("heldout ST labels stay locked until checkpoint selection")
    source_split = source / split
    target = output_root / split
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing ST split: {target}")
    minimum = int(lock["subtask_rt_dataset"]["minimum_phase_frames"])
    rows, audits = _analyze_split(source, split, programs, minimum, adjudications)
    fallback_rate = sum(row["fallback_used"] for row in audits) / len(audits)
    maximum = float(lock["subtask_rt_dataset"]["maximum_episode_fallback_rate"])
    if fallback_rate > maximum:
        raise ValueError(
            f"{split}: weak-label fallback rate {fallback_rate:.4f} exceeds {maximum:.4f}"
        )
    audit_by_episode = {int(row["episode_index"]): row for row in audits}
    program_indices, task_rows = _program_index(programs)
    building = output_root / f".{split}.building"
    if building.exists():
        raise FileExistsError(f"stale ST build directory requires audit: {building}")
    (building / "meta").mkdir(parents=True)
    output_episode_rows = []
    for row in rows:
        episode = int(row["episode_index"])
        source_parquet = source_split / f"data/chunk-000/episode_{episode:06d}.parquet"
        frame, states, actions = _episode_arrays(source_parquet)
        task_key = _source_task_key(row, programs)
        indices = program_indices[task_key]
        adjudication = adjudications.get((split, episode))
        segmentation = segment_subtasks(
            task_key,
            states,
            actions,
            skill_count=len(indices),
            minimum_phase_frames=minimum,
            boundary_override=(
                list(adjudication["consensus"]["boundaries"])
                if adjudication is not None
                else None
            ),
            boundary_override_audit=(adjudication["consensus"] if adjudication else None),
        )
        frame["task_index"] = np.asarray(indices, dtype=np.int64)[segmentation.labels]
        if "annotation.human.task_description" in frame:
            frame["annotation.human.task_description"] = frame["task_index"]
        destination_parquet = building / f"data/chunk-000/episode_{episode:06d}.parquet"
        destination_parquet.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(destination_parquet, index=False)
        source_video = (
            source_split
            / "videos/chunk-000/observation.images.front"
            / f"episode_{episode:06d}.mp4"
        )
        destination_video = (
            building
            / "videos/chunk-000/observation.images.front"
            / f"episode_{episode:06d}.mp4"
        )
        destination_video.parent.mkdir(parents=True, exist_ok=True)
        if not source_video.is_file():
            raise FileNotFoundError(source_video)
        os.symlink(source_video.resolve(), destination_video)
        program = next(
            item for item in programs["tasks"] if item["humanoidarena_task_key"] == task_key
        )
        output_episode_rows.append(
            {
                **row,
                "tasks": [skill["instruction"] for skill in program["skills"]],
                "subtask_boundary_audit": audit_by_episode[episode],
            }
        )

    info = load_json(source_split / "meta/info.json")
    info["total_tasks"] = len(task_rows)
    info["hrvla_subtask_relabel"] = {
        "schema_version": 1,
        "split": split,
        "method_program_sha256": canonical_sha256(programs),
        "training_only": True,
    }
    _json_atomic(building / "meta/info.json", info)
    shutil.copy2(source_split / "meta/modality.json", building / "meta/modality.json")
    shutil.copy2(source_split / "meta/stats.json", building / "meta/stats.json")
    _jsonl(building / "meta/tasks.jsonl", task_rows)
    _jsonl(building / "meta/episodes.jsonl", output_episode_rows)
    os.replace(building, target)
    core = {
        "schema_version": 1,
        "claim_boundary": "weak temporal labels for ST-RT training only; no closed-loop result",
        "split": split,
        "source_manifest_sha256": lock["source_dataset"]["manifest_sha256"],
        "method_program_sha256": canonical_sha256(programs),
        "episodes": len(rows),
        "frames": sum(int(row["length"]) for row in rows),
        "task_labels": len(task_rows),
        "minimum_phase_frames": minimum,
        "episode_fallback_rate": fallback_rate,
        "maximum_episode_fallback_rate": maximum,
        "episode_audits": audits,
    }
    manifest = {**core, "manifest_sha256": canonical_sha256(core)}
    manifest_path = output_root / "manifests" / f"{split}.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to replace ST manifest: {manifest_path}")
    _json_atomic(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock",
        type=Path,
        default=ROOT / "config/humanoidarena-rt-training.lock.json",
    )
    parser.add_argument(
        "--method-programs",
        type=Path,
        default=ROOT / "benchmark/humanoidarena_method_programs.json",
    )
    parser.add_argument("--split", action="append", choices=("train", "validation"))
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--audit-plot", type=Path)
    args = parser.parse_args()
    lock = load_json(args.lock.resolve())
    programs = load_json(args.method_programs.resolve())
    if lock.get("schema_version") != 1:
        raise ValueError("RT training lock schema_version must be 1")
    if canonical_sha256(programs) != lock["method_program_sha256"]:
        raise ValueError("method programs differ from the RT training lock")
    source = (ROOT / lock["source_dataset"]["path"]).resolve()
    if _file_sha256(source / "split_manifest.json") != lock["source_dataset"]["manifest_sha256"]:
        raise ValueError("source dataset manifest differs from the RT training lock")
    output_root = (ROOT / lock["subtask_rt_dataset"]["path"]).resolve()
    adjudications = _adjudication_index(lock)
    splits = args.split or list(lock["subtask_rt_dataset"]["splits_materialized_before_selection"])
    if args.audit_only:
        split_reports = {}
        for split in splits:
            rows, audits = _analyze_split(
                source,
                split,
                programs,
                int(lock["subtask_rt_dataset"]["minimum_phase_frames"]),
                adjudications,
            )
            split_reports[split] = {
                "episodes": len(rows),
                "fallback_episodes": sum(row["fallback_used"] for row in audits),
                "fallback_rate": sum(row["fallback_used"] for row in audits) / len(audits),
                "by_task": {
                    task: {
                        "episodes": len(selected),
                        "fallback_episodes": sum(row["fallback_used"] for row in selected),
                    }
                    for task in sorted({row["task_key"] for row in audits})
                    if (selected := [row for row in audits if row["task_key"] == task])
                },
            }
        core = {
            "schema_version": 1,
            "claim_boundary": "training-label audit only; not a closed-loop subtask score",
            "source_manifest_sha256": lock["source_dataset"]["manifest_sha256"],
            "method_program_sha256": canonical_sha256(programs),
            "maximum_episode_fallback_rate": lock["subtask_rt_dataset"][
                "maximum_episode_fallback_rate"
            ],
            "splits": split_reports,
        }
        report = {**core, "audit_sha256": canonical_sha256(core)}
        if args.audit_output is not None:
            _json_atomic(args.audit_output.resolve(), report)
        if args.audit_plot is not None:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            task_names = sorted(
                {
                    task
                    for split_report in split_reports.values()
                    for task in split_report["by_task"]
                }
            )
            width = 0.36
            x = np.arange(len(task_names))
            figure, axis = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
            for offset, split in enumerate(splits):
                values = [
                    split_reports[split]["by_task"][task]["fallback_episodes"]
                    / split_reports[split]["by_task"][task]["episodes"]
                    for task in task_names
                ]
                axis.bar(
                    x + (offset - (len(splits) - 1) / 2) * width,
                    values,
                    width,
                    label=split,
                )
            axis.axhline(
                float(lock["subtask_rt_dataset"]["maximum_episode_fallback_rate"]),
                color="#c53030",
                linestyle="--",
                label="frozen maximum",
            )
            axis.set_xticks(x, [name.replace("_", "\n") for name in task_names])
            axis.set_ylim(0, 1)
            axis.set_ylabel("Episode fallback rate")
            axis.set_title("HumanoidArena ST weak-label admission audit")
            axis.grid(axis="y", alpha=0.25)
            axis.legend()
            args.audit_plot.resolve().parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(args.audit_plot.resolve(), dpi=180)
            plt.close(figure)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    reports = [
        materialize_split(source, output_root, split, programs, lock, adjudications)
        for split in splits
    ]
    print(
        json.dumps(
            {
                report["split"]: {
                    "manifest_sha256": report["manifest_sha256"],
                    "episodes": report["episodes"],
                    "fallback_rate": report["episode_fallback_rate"],
                }
                for report in reports
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
