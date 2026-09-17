#!/usr/bin/env python3
"""Resolve rejected ST weak labels with audited Cosmos temporal consensus.

This runner only examines episodes for which the frozen state/action proxy used
an explicit fallback.  Three staggered contact sheets are inferred
independently and an episode is usable only when all confidence, agreement,
and minimum-phase gates pass.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MAXIMUM_FORMAT_REPAIRS = 2
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hrvla_bench.plan import canonical_sha256, load_json  # noqa: E402
from hrvla_bench.subtask_video_adjudication import (  # noqa: E402
    contact_sheet_indices,
    infer_temporal_proposal_with_repair,
    temporal_consensus,
)
from prepare_humanoidarena_subtask_data import _analyze_split  # noqa: E402


def _json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _compute_vram_mib() -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return sum(int(line.strip()) for line in result.stdout.splitlines() if line.strip())


def _assert_resource_gate(lock: dict[str, Any]) -> None:
    maximum = int(lock["resource_exclusion"]["maximum_preexisting_compute_vram_mib"])
    observed = _compute_vram_mib()
    if observed > maximum:
        raise RuntimeError(
            f"pre-existing compute VRAM is {observed} MiB, above the frozen {maximum} MiB gate"
        )


def _contact_sheet(frames: np.ndarray, indices: np.ndarray):
    from PIL import Image, ImageDraw

    columns = 6
    tile_width, tile_height = 192, 144
    label_height = 22
    rows = int(np.ceil(len(frames) / columns))
    sheet = Image.new("RGB", (columns * tile_width, rows * (tile_height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for position, (frame, frame_index) in enumerate(zip(frames, indices, strict=True)):
        row, column = divmod(position, columns)
        tile = Image.fromarray(frame).resize((tile_width, tile_height))
        x, y = column * tile_width, row * (tile_height + label_height)
        sheet.paste(tile, (x, y + label_height))
        draw.rectangle((x, y, x + tile_width, y + label_height), fill="black")
        draw.text((x + 5, y + 4), f"F{int(frame_index):05d}", fill="white")
    payload = BytesIO()
    sheet.save(payload, format="PNG", optimize=True)
    import hashlib

    return sheet, hashlib.sha256(payload.getvalue()).hexdigest()


def _prompt(task: dict[str, Any], episode_frames: int, sample_indices: np.ndarray) -> str:
    skills = [skill["instruction"] for skill in task["skills"]]
    transitions = [
        {
            "boundary_after_skill": task["skills"][index]["id"],
            "next_skill": task["skills"][index + 1]["id"],
        }
        for index in range(len(skills) - 1)
    ]
    return (
        "The image is a chronological contact sheet from one humanoid demonstration. "
        "Each tile is labeled with its exact LOCAL episode frame Fxxxxx. Locate the ordered "
        "semantic transition from each skill to its successor using only visible evidence. "
        "A boundary_frame_index MUST equal one of the printed sampled frame labels, must be "
        "strictly increasing, and means the first sampled frame where the next skill is visibly "
        "underway. Do not infer simulator success or invisible state. If evidence is ambiguous, "
        "lower confidence. Return a final JSON object only with keys boundary_frame_indices "
        "(integer list), boundary_confidences (0..1 list), and visible_evidence (short string "
        "list), with one entry per transition.\n"
        f"GOAL: {task['goal_instruction']}\n"
        f"ORDERED_SKILLS: {json.dumps(skills)}\n"
        f"TRANSITIONS: {json.dumps(transitions, sort_keys=True)}\n"
        f"EPISODE_FRAMES: {episode_frames}\n"
        f"SAMPLED_LOCAL_FRAMES: {sample_indices.tolist()}"
    )


class CosmosContactSheetModel:
    def __init__(self, repo_id: str, revision: str) -> None:
        import torch
        import transformers

        transformers.set_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        self._torch = torch
        self._model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            repo_id,
            revision=revision,
            local_files_only=True,
            dtype=torch.float16,
            device_map="cuda",
            attn_implementation="sdpa",
        ).eval()
        self._processor = transformers.Qwen3VLProcessor.from_pretrained(
            repo_id,
            revision=revision,
            local_files_only=True,
        )
        torch.cuda.reset_peak_memory_stats()

    def infer(self, image: Any, prompt: str) -> str:
        conversation = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You audit temporal skill boundaries in robot videos. "
                            "Use visible evidence and emit strict JSON."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        inputs = self._processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        with self._torch.inference_mode():
            generated = self._model.generate(**inputs, do_sample=False, max_new_tokens=384)
        trimmed = generated[:, inputs.input_ids.shape[1] :]
        return self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    @property
    def peak_vram_mib(self) -> int:
        return int(self._torch.cuda.max_memory_allocated() / (1024**2))


def _load_resumed(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    if not path.is_file():
        return {}
    output: dict[tuple[str, int], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        key = (str(row["split"]), int(row["episode_index"]))
        if key in output:
            raise ValueError(f"duplicate resume row: {key}")
        output[key] = row
    return output


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _load_attempt_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    output: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        attempt_id = str(row["attempt_id"])
        if attempt_id in output:
            raise ValueError(f"duplicate temporal inference attempt: {attempt_id}")
        output.add(attempt_id)
    return output


def _plot_report(report: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = sorted({row["task_key"] for row in report["records"]})
    accepted = [
        sum(row["consensus"]["accepted"] for row in report["records"] if row["task_key"] == task)
        / sum(row["task_key"] == task for row in report["records"])
        for task in tasks
    ]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    bars = axes[0].bar(tasks, accepted, color="#2b6cb0")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Consensus acceptance rate")
    axes[0].set_title("Video adjudication admission")
    axes[0].bar_label(bars, fmt="%.3f")
    axes[0].grid(axis="y", alpha=0.25)
    colors = ("#2f855a", "#d69e2e")
    for boundary_index, color in enumerate(colors):
        values = [
            row["consensus"]["boundaries"][boundary_index] / row["episode_frames"]
            for row in report["records"]
            if row["consensus"]["accepted"]
        ]
        if values:
            axes[1].hist(values, bins=10, alpha=0.6, color=color, label=f"boundary {boundary_index + 1}")
    axes[1].set_xlabel("Normalized episode time")
    axes[1].set_ylabel("Episodes")
    axes[1].set_title("Accepted temporal boundaries")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


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
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--retry-rejected", action="store_true")
    parser.add_argument("--skip-resource-gate", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    lock = load_json(args.lock.resolve())
    programs = load_json(args.method_programs.resolve())
    policy = lock["subtask_rt_dataset"]["temporal_video_adjudication"]
    if int(policy["sampling_variants"]) != 3:
        raise ValueError("only the frozen three-variant consensus is implemented")
    if canonical_sha256(programs) != lock["method_program_sha256"]:
        raise ValueError("method programs differ from the RT training lock")
    if not args.skip_resource_gate:
        _assert_resource_gate(lock)

    source = (ROOT / lock["source_dataset"]["path"]).resolve()
    minimum = int(lock["subtask_rt_dataset"]["minimum_phase_frames"])
    program_by_task = {row["humanoidarena_task_key"]: row for row in programs["tasks"]}
    metadata: dict[tuple[str, int], dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    baseline_fallbacks: dict[str, int] = {}
    split_totals: dict[str, int] = {}
    for split in lock["subtask_rt_dataset"]["splits_materialized_before_selection"]:
        episode_rows, audits = _analyze_split(source, split, programs, minimum)
        split_totals[split] = len(audits)
        baseline_fallbacks[split] = sum(row["fallback_used"] for row in audits)
        for episode_row, audit in zip(episode_rows, audits, strict=True):
            key = (split, int(audit["episode_index"]))
            metadata[key] = episode_row
            if audit["fallback_used"] and audit["task_key"] in policy["eligible_tasks"]:
                candidates.append({"split": split, **audit})

    report_path = (ROOT / policy["report_path"]).resolve()
    raw_path = report_path.with_name(report_path.stem + "_raw.jsonl")
    attempts_path = report_path.with_name(report_path.stem + "_attempts.jsonl")
    plot_path = report_path.with_name(report_path.stem + ".png")
    examples = report_path.with_name(report_path.stem + "_examples")
    resumed = _load_resumed(raw_path)
    observed_attempt_ids = _load_attempt_ids(attempts_path)
    pending = [
        row
        for row in candidates
        if (row["split"], int(row["episode_index"])) not in resumed
        or (args.retry_rejected and not resumed[(row["split"], int(row["episode_index"]))]["consensus"]["accepted"])
    ]
    if args.max_episodes is not None:
        pending = pending[: args.max_episodes]
    if args.retry_rejected and any(
        (row["split"], int(row["episode_index"])) in resumed for row in pending
    ):
        raise ValueError("retry-rejected requires a new output path to preserve immutable raw evidence")

    model = None
    if pending:
        model = CosmosContactSheetModel(policy["model_repo_id"], policy["model_revision"])
        from gr00t.utils.video_utils import get_frames_by_indices

        for candidate in pending:
            split = str(candidate["split"])
            episode = int(candidate["episode_index"])
            row = metadata[(split, episode)]
            length = int(row["length"])
            program = program_by_task[candidate["task_key"]]
            variant_rows = []
            proposals = []
            invalid_variants = []
            for variant in range(3):
                local_indices = contact_sheet_indices(
                    length, int(policy["sample_frames_per_variant"]), variant
                )
                packed_indices = local_indices + int(row["video_frame_offset"])
                frames = get_frames_by_indices(
                    str(row["source_video"]), packed_indices, decoder_kwargs={}
                )
                sheet, sheet_sha256 = _contact_sheet(frames, local_indices)
                if not any(
                    item["task_key"] == candidate["task_key"] and item["split"] == split
                    for item in resumed.values()
                ) and variant == 1:
                    examples.mkdir(parents=True, exist_ok=True)
                    sheet.save(examples / f"{split}-{candidate['task_key']}-episode-{episode:06d}.png")
                attempt_ids: list[str] = []

                def record_attempt(attempt: dict[str, Any]) -> None:
                    core = {
                        "schema_version": 1,
                        "split": split,
                        "episode_index": episode,
                        "task_key": candidate["task_key"],
                        "variant": variant,
                        "contact_sheet_sha256": sheet_sha256,
                        "model_repo_id": policy["model_repo_id"],
                        "model_revision": policy["model_revision"],
                        **attempt,
                    }
                    attempt_id = canonical_sha256(core)
                    attempt_ids.append(attempt_id)
                    if attempt_id not in observed_attempt_ids:
                        _append_jsonl(attempts_path, {**core, "attempt_id": attempt_id})
                        observed_attempt_ids.add(attempt_id)

                parsed, attempts = infer_temporal_proposal_with_repair(
                    lambda repair_prompt: model.infer(sheet, repair_prompt),
                    _prompt(program, length, local_indices),
                    expected_boundaries=len(program["skills"]) - 1,
                    episode_frames=length,
                    allowed_boundary_indices=local_indices,
                    maximum_format_repairs=MAXIMUM_FORMAT_REPAIRS,
                    record_attempt=record_attempt,
                )
                variant_row = {
                    "variant": variant,
                    "sample_indices": local_indices.tolist(),
                    "contact_sheet_sha256": sheet_sha256,
                    "attempt_ids": attempt_ids,
                    "attempt_count": len(attempts),
                    "format_repairs_used": len(attempts) - 1,
                    "raw_text": attempts[-1]["raw_text"],
                }
                if parsed is None:
                    invalid_variants.append(variant)
                    variant_rows.append(
                        {
                            **variant_row,
                            "valid": False,
                            "parse_error": attempts[-1]["parse_error"],
                        }
                    )
                else:
                    proposals.append(parsed)
                    variant_rows.append(
                        {
                            **variant_row,
                            "valid": True,
                            "parse_error": None,
                            "boundaries": list(parsed.boundaries),
                            "confidences": list(parsed.confidences),
                            "visible_evidence": list(parsed.evidence),
                        }
                    )
            consensus = temporal_consensus(
                proposals,
                episode_frames=length,
                skills=len(program["skills"]),
                minimum_phase_frames=minimum,
                minimum_confidence=float(policy["minimum_boundary_confidence"]),
                maximum_spread_fraction=float(policy["maximum_boundary_spread_fraction"]),
            )
            if invalid_variants:
                consensus["reasons"] = sorted(
                    set(consensus["reasons"])
                    | {"invalid_model_output_after_bounded_format_repair"}
                )
                consensus["invalid_variants"] = invalid_variants
            result = {
                "schema_version": 2,
                "split": split,
                "episode_index": episode,
                "source_dataset": row["source_dataset"],
                "source_episode_index": int(row["source_episode_index"]),
                "source_video": row["source_video"],
                "video_frame_offset": int(row["video_frame_offset"]),
                "episode_frames": length,
                "task_key": candidate["task_key"],
                "state_action_proxy": candidate["proxy"],
                "variants": variant_rows,
                "consensus": consensus,
            }
            _append_jsonl(raw_path, result)
            resumed[(split, episode)] = result
            print(
                json.dumps(
                    {"split": split, "episode": episode, "task": candidate["task_key"], "consensus": consensus},
                    sort_keys=True,
                ),
                flush=True,
            )

    all_candidate_keys = {(row["split"], int(row["episode_index"])) for row in candidates}
    complete = all_candidate_keys <= set(resumed)
    records = [resumed[key] for key in sorted(all_candidate_keys & set(resumed))]
    accepted_by_split = {
        split: sum(row["split"] == split and row["consensus"]["accepted"] for row in records)
        for split in split_totals
    }
    residual_rates = {
        split: (baseline_fallbacks[split] - accepted_by_split[split]) / split_totals[split]
        for split in split_totals
    }
    maximum = float(lock["subtask_rt_dataset"]["maximum_episode_fallback_rate"])
    status = "pass" if complete and all(rate <= maximum for rate in residual_rates.values()) else "fail"
    core = {
        "schema_version": 2,
        "status": status,
        "claim_boundary": "training-label adjudication only; not a closed-loop success metric",
        "source_manifest_sha256": lock["source_dataset"]["manifest_sha256"],
        "method_program_sha256": lock["method_program_sha256"],
        "model_repo_id": policy["model_repo_id"],
        "model_revision": policy["model_revision"],
        "policy": {key: policy[key] for key in (
            "eligible_tasks",
            "sample_frames_per_variant",
            "sampling_variants",
            "minimum_boundary_confidence",
            "maximum_boundary_spread_fraction",
        )},
        "inference_protocol": {
            "maximum_format_repairs": MAXIMUM_FORMAT_REPAIRS,
            "invalid_output_policy": "reject_episode_and_continue",
            "repair_scope": "structure_only_no_local_boundary_modification",
        },
        "complete": complete,
        "candidates_expected": len(candidates),
        "candidates_observed": len(records),
        "accepted": sum(row["consensus"]["accepted"] for row in records),
        "baseline_fallbacks": baseline_fallbacks,
        "accepted_by_split": accepted_by_split,
        "residual_fallback_rates": residual_rates,
        "maximum_episode_fallback_rate": maximum,
        "peak_vram_mib": model.peak_vram_mib if model is not None else None,
        "raw_evidence_path": str(raw_path.relative_to(ROOT)),
        "raw_attempt_evidence_path": str(attempts_path.relative_to(ROOT)),
        "records": [
            {key: value for key, value in row.items() if key != "variants"}
            | {
                "variant_evidence": [
                    {key: value for key, value in variant.items() if key != "raw_text"}
                    for variant in row["variants"]
                ]
            }
            for row in records
        ],
    }
    report = {**core, "audit_sha256": canonical_sha256(core)}
    _json_atomic(report_path, report)
    if records:
        _plot_report(report, plot_path)
    print(json.dumps({key: report[key] for key in (
        "status", "complete", "candidates_expected", "candidates_observed", "accepted", "residual_fallback_rates", "audit_sha256"
    )}, indent=2, sort_keys=True))
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
