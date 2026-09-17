#!/usr/bin/env python3
"""Stage, publish, and compile immutable HumanoidArena benchmark artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrvla_bench.humanoidarena_release import (  # noqa: E402
    build_release_manifest,
    file_sha256,
    git_revision,
    release_receipt,
    runtime_signatures,
    validate_release_receipt,
    verify_release_manifest,
    write_json_once,
)
from hrvla_bench.humanoidarena_rt_training import (  # noqa: E402
    RT_METHODS,
    load_rt_lock,
    run_directory,
    select_checkpoint,
    validate_rt_inputs,
)
from hrvla_bench.humanoidarena_training import (  # noqa: E402
    load_training_lock,
    seed_directory,
    validate_selection,
)
from hrvla_bench.internal_matrix import validate_ready_checkpoint_lock  # noqa: E402
from hrvla_bench.plan import canonical_sha256, load_json  # noqa: E402


DEFAULT_ROOT = ROOT / "_artifacts/HumanoidArena/release/humanoidarena-v1"
DEFAULT_PLAN = DEFAULT_ROOT / "release-plan.json"
DEFAULT_CANDIDATE_LOCK = DEFAULT_ROOT / "humanoidarena-internal-checkpoints.lock.json"
DEFAULT_PROBE = DEFAULT_ROOT / "coexistence-probe.json"
RELEASE_FAMILIES = ("common", "subtask_rt", "recovery_rt")


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _load_selection(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _tracked_clean() -> bool:
    result = subprocess.run(
        ["git", "status", "--short", "--untracked-files=no"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return not result.stdout.strip()


def _validate_rt_selection(
    lock: dict[str, Any], method_id: str, selection: dict[str, Any]
) -> int:
    expected = select_checkpoint(ROOT, lock, method_id)
    if expected != selection:
        raise ValueError(f"{method_id}: frozen selection differs from validation evidence")
    step = selection.get("selected_step")
    if type(step) is not int or step not in lock["training"]["candidate_steps"]:
        raise ValueError(f"{method_id}: selected step is not locked")
    return step


def _dataset_provenance(lock: dict[str, Any], family: str) -> dict[str, Any]:
    config = lock[family]
    manifests = {}
    for split in ("train", "validation"):
        expected = config["manifests"].get(split)
        path = ROOT / config["path"] / "manifests" / f"{split}.json"
        value = load_json(path)
        core = {key: item for key, item in value.items() if key != "manifest_sha256"}
        if not isinstance(expected, str) or canonical_sha256(core) != expected:
            raise ValueError(f"{family}/{split}: dataset manifest is not frozen")
        manifests[split] = expected
    return {"dataset_family": family, "manifests": manifests}


def _artifact_specifications(
    common_lock: dict[str, Any],
    rt_lock: dict[str, Any],
    families: tuple[str, ...] = RELEASE_FAMILIES,
) -> list[dict[str, Any]]:
    if not families or len(set(families)) != len(families) or any(
        family not in RELEASE_FAMILIES for family in families
    ):
        raise ValueError("release families must be a non-empty unique locked subset")
    common_selection_path = ROOT / common_lock["output_root"] / "selection.json"
    common_selection = _load_selection(common_selection_path)
    common_step = validate_selection(common_selection, common_lock)
    requested_methods = tuple(
        method_id
        for family, method_id in (
            ("subtask_rt", "gr00t_st_rt"),
            ("recovery_rt", "gr00t_str_rt"),
        )
        if family in families
    )
    if requested_methods:
        validate_rt_inputs(
            ROOT, rt_lock, common_lock, common_selection, requested_methods
        )
    rt_selections = {}
    rt_steps = {}
    for method_id in requested_methods:
        name = rt_lock["methods"][method_id]["output_name"]
        path = ROOT / rt_lock["output_root"] / name / "selection.json"
        selection = _load_selection(path)
        rt_steps[method_id] = _validate_rt_selection(rt_lock, method_id, selection)
        rt_selections[method_id] = selection

    specs = []
    if "common" in families:
        for seed in common_lock["training_seeds"]:
            root = seed_directory(ROOT, common_lock, seed) / f"checkpoints/checkpoint-{common_step}"
            specs.append(
                {
                    "artifact_id": f"common-seed-{seed}",
                    "artifact_type": "checkpoint",
                    "root": _relative(root),
                    "repo_type": "model",
                    "path_in_repo": f"humanoidarena-v1/checkpoints/common/seed-{seed}/checkpoint-{common_step}",
                    "provenance": {
                        "family": "common",
                        "training_seed": seed,
                        "selected_step": common_step,
                        "selection_sha256": common_selection["selection_sha256"],
                    },
                },
            )
    for method_id in requested_methods:
        family = "subtask_rt" if method_id == "gr00t_st_rt" else "recovery_rt"
        step = rt_steps[method_id]
        for seed in rt_lock["training_seeds"]:
            root = run_directory(ROOT, rt_lock, method_id, seed) / f"checkpoints/checkpoint-{step}"
            specs.append(
                {
                    "artifact_id": f"{family}-seed-{seed}",
                    "artifact_type": "checkpoint",
                    "root": _relative(root),
                    "repo_type": "model",
                    "path_in_repo": f"humanoidarena-v1/checkpoints/{family}/seed-{seed}/checkpoint-{step}",
                    "provenance": {
                        "family": family,
                        "method_id": method_id,
                        "training_seed": seed,
                        "selected_step": step,
                        "selection_sha256": rt_selections[method_id]["selection_sha256"],
                    },
                }
            )
    for release_family, dataset_family, name in (
        ("subtask_rt", "subtask_rt_dataset", "subtask-rt-dataset"),
        ("recovery_rt", "recovery_rt_dataset", "recovery-rt-dataset"),
    ):
        if release_family not in families:
            continue
        specs.append(
            {
                "artifact_id": name,
                "artifact_type": "dataset",
                "root": rt_lock[dataset_family]["path"],
                "repo_type": "dataset",
                "path_in_repo": f"humanoidarena-v1/{name}",
                "provenance": _dataset_provenance(rt_lock, dataset_family),
            }
        )
    return specs


def stage(args: argparse.Namespace) -> int:
    if not _tracked_clean():
        raise RuntimeError("release staging requires a clean tracked worktree")
    common_lock = load_training_lock(args.common_lock.resolve())
    rt_lock = load_rt_lock(args.rt_lock.resolve())
    revision = git_revision(ROOT)
    plan_root = args.plan.resolve().parent
    families = tuple(args.family or RELEASE_FAMILIES)
    rows = []
    for spec in _artifact_specifications(common_lock, rt_lock, families):
        repo_id = args.model_repo if spec["repo_type"] == "model" else args.dataset_repo
        manifest = build_release_manifest(
            ROOT,
            artifact_id=spec["artifact_id"],
            artifact_type=spec["artifact_type"],
            root=spec["root"],
            repo_id=repo_id,
            repo_type=spec["repo_type"],
            path_in_repo=spec["path_in_repo"],
            source_revision=revision,
            provenance=spec["provenance"],
        )
        manifest_path = plan_root / "manifests" / f"{spec['artifact_id']}.json"
        receipt_path = plan_root / "receipts" / f"{spec['artifact_id']}.json"
        write_json_once(manifest_path, manifest)
        rows.append(
            {
                "artifact_id": spec["artifact_id"],
                "manifest_path": _relative(manifest_path),
                "manifest_sha256": manifest["manifest_sha256"],
                "receipt_path": _relative(receipt_path),
            }
        )
    core = {
        "schema_version": 1,
        "status": (
            "staged_waiting_for_immutable_publication"
            if families == RELEASE_FAMILIES
            else "partial_staged_waiting_for_immutable_publication"
        ),
        "source_revision": revision,
        "families": list(families),
        "artifacts": rows,
    }
    plan = {**core, "release_plan_sha256": canonical_sha256(core)}
    write_json_once(args.plan.resolve(), plan)
    print(
        json.dumps(
            {
                "artifacts": len(rows),
                "families": list(families),
                "release_plan_sha256": plan["release_plan_sha256"],
            }
        )
    )
    return 0


def _load_plan(path: Path) -> dict[str, Any]:
    plan = load_json(path)
    core = {key: value for key, value in plan.items() if key != "release_plan_sha256"}
    if canonical_sha256(core) != plan.get("release_plan_sha256"):
        raise ValueError("release plan hash differs")
    return plan


def _load_manifest(row: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = ROOT / row["manifest_path"]
    manifest = load_json(path)
    if manifest.get("manifest_sha256") != row.get("manifest_sha256"):
        raise ValueError(f"{row['artifact_id']}: plan and manifest hashes differ")
    errors = verify_release_manifest(manifest, ROOT)
    if errors:
        raise ValueError(f"{row['artifact_id']}: " + "; ".join(errors))
    return path, manifest


def publish(args: argparse.Namespace) -> int:
    try:
        from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    except ImportError as error:
        raise RuntimeError("publish requires huggingface_hub") from error
    plan = _load_plan(args.plan.resolve())
    selected = set(args.artifact or [])
    unknown = selected - {row["artifact_id"] for row in plan["artifacts"]}
    if unknown:
        raise ValueError(f"unknown release artifacts: {sorted(unknown)}")
    api = HfApi()
    for row in plan["artifacts"]:
        if selected and row["artifact_id"] not in selected:
            continue
        manifest_path, manifest = _load_manifest(row)
        receipt_path = ROOT / row["receipt_path"]
        if receipt_path.is_file():
            validate_release_receipt(load_json(receipt_path), manifest)
            print(f"skip published {row['artifact_id']}")
            continue
        api.create_repo(
            manifest["repo_id"],
            repo_type=manifest["repo_type"],
            private=args.private,
            exist_ok=True,
        )
        prefix = manifest["path_in_repo"].rstrip("/")
        existing = api.list_repo_files(manifest["repo_id"], repo_type=manifest["repo_type"])
        if any(path == prefix or path.startswith(prefix + "/") for path in existing):
            raise FileExistsError(
                f"remote path already exists without a local verified receipt: {prefix}"
            )
        root = ROOT / manifest["root"]
        operations = [
            CommitOperationAdd(
                path_in_repo=f"{prefix}/{item['path']}",
                path_or_fileobj=root / item["path"],
            )
            for item in manifest["files"]
        ]
        remote_manifest_path = f"{prefix}/hrvla-release-manifest.json"
        operations.append(
            CommitOperationAdd(
                path_in_repo=remote_manifest_path,
                path_or_fileobj=manifest_path,
            )
        )
        info = api.create_commit(
            repo_id=manifest["repo_id"],
            repo_type=manifest["repo_type"],
            operations=operations,
            commit_message=f"feat(HumanoidArena): publish {row['artifact_id']}",
        )
        revision = str(info.oid)
        downloaded = Path(
            hf_hub_download(
                manifest["repo_id"],
                remote_manifest_path,
                repo_type=manifest["repo_type"],
                revision=revision,
            )
        )
        if load_json(downloaded) != manifest or file_sha256(downloaded) != file_sha256(manifest_path):
            raise ValueError(f"remote release manifest verification failed: {row['artifact_id']}")
        receipt = release_receipt(
            manifest,
            published_revision=revision,
            remote_manifest_path=remote_manifest_path,
        )
        write_json_once(receipt_path, receipt)
        print(json.dumps({"artifact_id": row["artifact_id"], "published_revision": revision}))
    return 0


def compile_lock(args: argparse.Namespace) -> int:
    plan = _load_plan(args.plan.resolve())
    manifests = {}
    receipts = {}
    for row in plan["artifacts"]:
        _path, manifest = _load_manifest(row)
        receipt_path = ROOT / row["receipt_path"]
        receipt = load_json(receipt_path)
        validate_release_receipt(receipt, manifest)
        manifests[row["artifact_id"]] = manifest
        receipts[row["artifact_id"]] = receipt
    template = load_json(args.template.resolve())
    probe = load_json(args.probe.resolve())
    limit = template["coexistence_probe"]["maximum_peak_compute_vram_mib"]
    if probe.get("status") not in {"passed", "failed_cpu_fallback"}:
        raise ValueError("coexistence probe status is unresolved")
    if probe.get("policy_device") not in {"cpu", "cuda:0"}:
        raise ValueError("coexistence probe policy_device is unresolved")
    if type(probe.get("measured_peak_compute_vram_mib")) not in {int, float}:
        raise ValueError("coexistence probe peak VRAM is missing")
    common_probe_manifest = manifests["common-seed-0"]
    if probe.get("checkpoint_manifest_sha256") != common_probe_manifest["manifest_sha256"]:
        raise ValueError("coexistence probe used a different common checkpoint manifest")
    if probe.get("checkpoint_path") != common_probe_manifest["root"]:
        raise ValueError("coexistence probe used a different common checkpoint path")

    def checkpoint_row(artifact_id: str) -> dict[str, Any]:
        manifest = manifests[artifact_id]
        receipt = receipts[artifact_id]
        provenance = manifest["provenance"]
        return {
            "checkpoint_id": (
                f"hf://{manifest['repo_id']}@{receipt['published_revision']}/"
                f"{manifest['path_in_repo']}"
            ),
            "path": manifest["root"],
            "manifest_path": next(
                row["manifest_path"] for row in plan["artifacts"] if row["artifact_id"] == artifact_id
            ),
            "published_revision": receipt["published_revision"],
            "manifest_sha256": manifest["manifest_sha256"],
            "training_seed": provenance["training_seed"],
            "selected_step": provenance["selected_step"],
            "selection_sha256": provenance["selection_sha256"],
        }

    checkpoints = {}
    common = {str(seed): checkpoint_row(f"common-seed-{seed}") for seed in (0, 1, 2)}
    for method_id in ("gr00t_sonic", "gr00t_st", "gr00t_str"):
        checkpoints[method_id] = common
    checkpoints["gr00t_st_rt"] = {
        str(seed): checkpoint_row(f"subtask_rt-seed-{seed}") for seed in (0, 1, 2)
    }
    checkpoints["gr00t_str_rt"] = {
        str(seed): checkpoint_row(f"recovery_rt-seed-{seed}") for seed in (0, 1, 2)
    }
    candidate = {
        **template,
        "status": "ready_for_frozen_execution",
        "claim_boundary": "Immutable published HumanoidArena checkpoints validated for the frozen internal matrix.",
        "method_runtime_signatures": runtime_signatures(ROOT),
        "policy_device": probe["policy_device"],
        "coexistence_probe": {
            **template["coexistence_probe"],
            **probe,
            "maximum_peak_compute_vram_mib": limit,
        },
        "checkpoint_families": {
            "common": {**template["checkpoint_families"]["common"], "status": "published_verified"},
            "subtask_rt": {**template["checkpoint_families"]["subtask_rt"], "status": "published_verified"},
            "recovery_rt": {**template["checkpoint_families"]["recovery_rt"], "status": "published_verified"},
        },
        "release_plan_sha256": plan["release_plan_sha256"],
        "dataset_release_receipts": {
            artifact_id: receipts[artifact_id]
            for artifact_id in ("subtask-rt-dataset", "recovery-rt-dataset")
        },
        "checkpoints": checkpoints,
    }
    validate_ready_checkpoint_lock(
        candidate,
        method_program_sha256=candidate["method_program_sha256"],
        internal_protocol_sha256=candidate["internal_protocol_sha256"],
    )
    write_json_once(args.output.resolve(), candidate)
    print(json.dumps({"output": str(args.output), "checkpoints": 9, "policy_device": probe["policy_device"]}))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    common = subparsers.add_parser("stage")
    common.add_argument("--common-lock", type=Path, default=ROOT / "config/humanoidarena-gr00t-training.lock.json")
    common.add_argument("--rt-lock", type=Path, default=ROOT / "config/humanoidarena-rt-training.lock.json")
    common.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    common.add_argument("--model-repo", default="shin0412/HRVLA")
    common.add_argument("--dataset-repo", default="shin0412/HRVLA")
    common.add_argument("--family", action="append", choices=RELEASE_FAMILIES)
    common.set_defaults(handler=stage)
    publishing = subparsers.add_parser("publish")
    publishing.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    publishing.add_argument("--artifact", action="append")
    publishing.add_argument("--private", action="store_true")
    publishing.set_defaults(handler=publish)
    compiling = subparsers.add_parser("compile-lock")
    compiling.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    compiling.add_argument("--template", type=Path, default=ROOT / "config/humanoidarena-internal-checkpoints.lock.json")
    compiling.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    compiling.add_argument("--output", type=Path, default=DEFAULT_CANDIDATE_LOCK)
    compiling.set_defaults(handler=compile_lock)
    return result


def main() -> int:
    args = parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
