"""Admission checks for the locked HumanoidArena benchmark release."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any


def _check(check_id: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"id": check_id, "status": "pass" if ok else "fail", "detail": detail}


def _git_revision(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _load_oracle(path: Path, required_trials: int, required_successes: int) -> tuple[bool, str]:
    if not path.is_file():
        return False, f"missing oracle evidence: {path}"
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
        trials = int(evidence["trials"])
        successes = int(evidence["successes"])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return False, f"invalid oracle evidence: {exc}"
    ok = trials == required_trials and successes == required_successes
    required = f"{required_successes}/{required_trials}"
    return ok, f"oracle successes: {successes}/{trials} (required: {required})"


def admission_report(
    lock: dict[str, Any],
    repo_root: Path,
    *,
    source_root: Path | None = None,
    model_root: Path | None = None,
    dataset_root: Path | None = None,
    asset_root: Path | None = None,
    sonic_policy_root: Path | None = None,
    oracle_evidence: Path | None = None,
    isaac_sim_version: str | None = None,
    isaac_lab_ref: str | None = None,
) -> dict[str, Any]:
    """Return a machine-readable report; no check downloads or mutates artifacts."""
    source = lock["source"]
    resources = lock["resources"]
    runtime = lock["runtime"]

    def resolve(value: Path | None, default: str) -> Path:
        path = value if value is not None else Path(default)
        return path if path.is_absolute() else repo_root / path

    source_path = resolve(source_root, source["default_local_path"])
    models_path = resolve(model_root, resources["models"]["default_local_path"])
    dataset_path = resolve(dataset_root, resources["dataset"]["default_local_path"])
    assets_path = resolve(asset_root, resources["assets"]["default_local_path"])
    sonic_path = resolve(sonic_policy_root, resources["sonic_policy"]["default_local_path"])
    oracle_path = resolve(oracle_evidence, lock["admission"]["oracle_evidence_default_path"])

    checks: list[dict[str, Any]] = []
    revision = _git_revision(source_path)
    checks.append(
        _check(
            "source_revision",
            revision == source["revision"],
            f"actual={revision or 'missing'} expected={source['revision']}",
        )
    )
    for relative in lock["required_source_paths"]:
        checks.append(
            _check(f"source_path:{relative}", (source_path / relative).is_file(), relative)
        )
    for task in lock["tasks"]:
        relative = f"isaaclab_twist2_g1/batch_test_scripts/task/batch_test_{task}.sh"
        checks.append(_check(f"task_script:{task}", (source_path / relative).is_file(), relative))
    for mode in lock["evaluation_modes"]:
        relative = f"isaaclab_twist2_g1/tasks/common_test_config/{mode}"
        checks.append(
            _check(f"evaluation_mode:{mode}", (source_path / relative).is_dir(), relative)
        )
    self_test = lock["upstream_self_test"]
    checks.append(
        _check(
            "upstream_self_test",
            self_test["status_at_lock"] == "passing",
            f"status={self_test['status_at_lock']} command={self_test['command']}",
        )
    )

    checks.append(
        _check(
            "isaac_sim_version",
            isaac_sim_version == runtime["isaac_sim"],
            f"actual={isaac_sim_version or 'unreported'} expected={runtime['isaac_sim']}",
        )
    )
    checks.append(
        _check(
            "isaac_lab_ref",
            isaac_lab_ref == runtime["isaac_lab_ref"],
            f"actual={isaac_lab_ref or 'unreported'} expected={runtime['isaac_lab_ref']}",
        )
    )
    for directory in resources["assets"]["required_directories"]:
        checks.append(
            _check(
                f"asset:{directory}",
                (assets_path / directory).is_dir(),
                str(assets_path / directory),
            )
        )
    for filename in resources["sonic_policy"]["required_files"]:
        checks.append(
            _check(
                f"sonic_policy:{filename}",
                (sonic_path / filename).is_file(),
                str(sonic_path / filename),
            )
        )

    model_tasks = {
        "boxing": "HSI_boxing",
        "doubledesk": "HOI_double_desk",
        "football": "HOI_football",
        "open_door": "HSI_open_door",
        "pp_box": "HOI_pp_box",
        "sit_sofa": "HSI_sit_sofa",
        "vision_navi": "HSI_vision_navi",
    }
    for task, directory in model_tasks.items():
        has_checkpoint = any((models_path / "pi" / directory).glob("**/model.safetensors"))
        checks.append(
            _check(f"released_model:{task}", has_checkpoint, str(models_path / "pi" / directory))
        )
    for task, directory in model_tasks.items():
        info = dataset_path / directory / "sonic_refpose_v3_1/meta/info.json"
        checks.append(_check(f"released_dataset:{task}", info.is_file(), str(info)))

    required_trials = int(lock["admission"]["required_oracle_trials"])
    required_successes = int(lock["admission"]["required_oracle_successes"])
    oracle_ok, oracle_detail = _load_oracle(
        oracle_path, required_trials, required_successes
    )
    checks.append(_check("oracle_admission", oracle_ok, oracle_detail))
    blockers = [item["id"] for item in checks if item["status"] != "pass"]
    return {
        "schema_version": 1,
        "benchmark": "HumanoidArena",
        "source_revision": source["revision"],
        "model_revision": resources["models"]["revision"],
        "dataset_revision": resources["dataset"]["revision"],
        "admitted": not blockers,
        "checks": checks,
        "blockers": blockers,
    }
