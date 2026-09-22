#!/usr/bin/env python3
"""Pair audited PI0.5-ST HA episodes with the frozen PI0.5 baseline."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import random
import shutil


ROOT = Path(__file__).resolve().parents[1]
LOCKS = ROOT / "benchmark/locks/pi05_ha"
TASK_LABELS = {
    "boxing": "Boxing",
    "doubledesk": "DoubleDesk",
    "football": "Football",
    "pp_box": "PickPlaceBox",
    "sit_sofa": "SitSofa",
    "vision_navi": "VisionNavigation",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"evidence path escapes runtime root: {relative}")
    return path


def _key(row: dict) -> tuple[str, str, int, int, int]:
    return (
        row["task"], row["mode"], int(row["seed"]),
        int(row["repeat_idx"]), int(row["episode_seed"]),
    )


def _bootstrap_delta(pairs: list[dict], repetitions: int = 10000) -> list[float]:
    by_task = {
        task: [int(row["st_success"]) - int(row["baseline_success"])
               for row in pairs if row["task"] == task]
        for task in TASK_LABELS
    }
    rng = random.Random(20260922)
    estimates = []
    for _ in range(repetitions):
        total = sum(
            values[rng.randrange(len(values))]
            for values in by_task.values()
            for _ in range(len(values))
        )
        estimates.append(total / len(pairs))
    estimates.sort()
    return [estimates[int(0.025 * repetitions)], estimates[int(0.975 * repetitions) - 1]]


def _svg(by_task: dict, overall: dict) -> str:
    labels = list(TASK_LABELS) + ["overall"]
    height = 82 + 66 * len(labels)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}" viewBox="0 0 1000 {height}">',
        '<rect width="1000" height="100%" fill="#ffffff"/>',
        '<text x="28" y="30" font-family="sans-serif" font-size="21" fill="#172033">PI0.5 + SONIC vs PI0.5-ST · HumanoidArena</text>',
        '<text x="28" y="52" font-family="sans-serif" font-size="13" fill="#475569">Same 1,440 task × mode × seed × repeat identities; OpenDoor excluded</text>',
    ]
    for index, task in enumerate(labels):
        row = overall if task == "overall" else by_task[task]
        y = 84 + 66 * index
        title = "Six-task primary" if task == "overall" else TASK_LABELS[task]
        lines.append(
            f'<text x="28" y="{y + 19}" font-family="sans-serif" font-size="15" fill="#172033">{html.escape(title)}</text>'
        )
        for offset, key, color, caption in (
            (0, "baseline", "#64748b", "base"),
            (25, "st", "#0d9488", "ST"),
        ):
            rate = row[f"{key}_success_rate"]
            width = int(rate * 540)
            lines.append(
                f'<rect x="242" y="{y + offset}" width="{width}" height="18" rx="3" fill="{color}"/>'
            )
            lines.append(
                f'<text x="{max(248, 252 + width)}" y="{y + offset + 14}" '
                f'font-family="sans-serif" font-size="13" fill="#172033">'
                f'{caption}: {row[f"{key}_successes"]}/{row["episodes"]} ({100 * rate:.1f}%)</text>'
            )
    lines.append('</svg>')
    return "\n".join(lines) + "\n"


def compile_comparison(runtime_root: Path, primary_audit: Path, output_root: Path) -> dict:
    protocol = json.loads((LOCKS / "protocol.json").read_text(encoding="utf-8"))
    baseline_path = runtime_root / "results/benchmark/external/pi05_cuda_int8_v1_amended_summary.json"
    if sha256(baseline_path) != protocol["amended_summary_sha256"]:
        raise ValueError("baseline selected-source manifest differs from locked summary")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    st = json.loads(primary_audit.read_text(encoding="utf-8"))
    if (
        st.get("audit_passed") is not True
        or st.get("method_id") != "pi05_st"
        or st.get("diagnostic") is not False
        or st.get("episodes_observed") != 1440
    ):
        raise ValueError("PI0.5-ST primary audit is not a complete paired result")
    baseline_rows = {}
    for record in baseline["selected_episode_manifest"]:
        if record["task"] not in TASK_LABELS:
            continue
        path = _inside(runtime_root, record["path"])
        if sha256(path) != record["sha256"]:
            raise ValueError(f"baseline raw row hash differs: {path}")
        episode = json.loads(path.read_text(encoding="utf-8"))
        identity = {
            "task": record["task"], "mode": record["mode"],
            "seed": record["seed"], "repeat_idx": record["repeat_idx"],
            "episode_seed": episode["episode_seed"],
        }
        key = _key(identity)
        if key in baseline_rows:
            raise ValueError(f"duplicate baseline episode identity: {key}")
        baseline_rows[key] = {
            "success": episode["success"], "path": record["path"],
            "sha256": record["sha256"],
        }
    st_rows = {}
    for row in st["episode_manifest"]:
        key = _key(row)
        if key in st_rows:
            raise ValueError(f"duplicate ST episode identity: {key}")
        st_rows[key] = row
    if len(baseline_rows) != 1440 or set(baseline_rows) != set(st_rows):
        raise ValueError("PI0.5-ST and baseline are not paired on all 1,440 identities")
    pairs = [
        {
            "task": key[0], "mode": key[1], "seed": key[2],
            "repeat_idx": key[3], "episode_seed": key[4],
            "baseline_success": baseline_rows[key]["success"],
            "st_success": st_rows[key]["success"],
            "baseline_episode_sha256": baseline_rows[key]["sha256"],
            "st_episode_sha256": st_rows[key]["episode_sha256"],
        }
        for key in sorted(baseline_rows)
    ]

    def summary(rows: list[dict]) -> dict:
        base = sum(row["baseline_success"] for row in rows)
        candidate = sum(row["st_success"] for row in rows)
        discordant = Counter(
            (int(row["baseline_success"]), int(row["st_success"]))
            for row in rows
        )
        return {
            "episodes": len(rows),
            "baseline_successes": base,
            "baseline_success_rate": base / len(rows),
            "st_successes": candidate,
            "st_success_rate": candidate / len(rows),
            "st_minus_baseline_success_rate": (candidate - base) / len(rows),
            "baseline_only_successes": discordant[(1, 0)],
            "st_only_successes": discordant[(0, 1)],
            "both_successes": discordant[(1, 1)],
            "neither_successes": discordant[(0, 0)],
        }

    by_task = {
        task: summary([row for row in pairs if row["task"] == task])
        for task in TASK_LABELS
    }
    overall = summary(pairs)
    overall["paired_stratified_bootstrap_delta_95"] = _bootstrap_delta(pairs)
    output_root.mkdir(parents=True, exist_ok=True)
    video_dir = output_root / "videos"
    video_dir.mkdir(exist_ok=True)
    video_manifest = []
    for task in TASK_LABELS:
        samples = [
            row for row in st["episode_manifest"]
            if row["task"] == task and row["video_sha256"]
        ]
        selected = min(samples, key=lambda row: (
            row["mode"], row["seed"], row["repeat_idx"]
        ))
        source = Path(selected["video_path"])
        if sha256(source) != selected["video_sha256"]:
            raise ValueError(f"ST sample video differs from audited row: {task}")
        target = video_dir / f"{task}.mp4"
        if target.exists() and sha256(target) != selected["video_sha256"]:
            raise ValueError(f"existing sample video differs: {target}")
        if not target.exists():
            shutil.copy2(source, target)
        video_manifest.append({
            "task": task, "path": str(target), "sha256": sha256(target),
            "selection": "lexicographic first sampled mode/seed/repeat, not outcome-selected",
        })
    result = {
        "schema_version": 1,
        "audit_passed": True,
        "claim_boundary": "paired PI0.5-ST versus PI0.5 HA six-task nominal result only",
        "episodes_observed": 1440,
        "baseline_summary_sha256": sha256(baseline_path),
        "st_primary_audit_sha256": sha256(primary_audit),
        "overall": overall,
        "by_task": by_task,
        "sample_video_manifest": video_manifest,
        "paired_episode_manifest": pairs,
    }
    (output_root / "comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "comparison.svg").write_text(_svg(by_task, overall), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--primary-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "comparison.json").exists():
        raise FileExistsError("refusing to overwrite a completed paired comparison")
    result = compile_comparison(
        args.runtime_root.resolve(strict=True),
        args.primary_audit.resolve(strict=True), output,
    )
    print(json.dumps({key: value for key, value in result.items() if key not in {"paired_episode_manifest", "sample_video_manifest"}}, indent=2))


if __name__ == "__main__":
    main()
