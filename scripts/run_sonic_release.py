#!/usr/bin/env python3
"""Run the locked NVIDIA SONIC release in its isolated Isaac Lab runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = REPO_ROOT / "config" / "sonic-release.lock.json"
VENDOR_ROOT = REPO_ROOT / "_vendor"
SONIC_ROOT = VENDOR_ROOT / "GR00T-WholeBodyControl"
RUNTIME_PYTHON = VENDOR_ROOT / "sonic-runtime" / "bin" / "python"
ISAACLAB_CANDIDATES = (VENDOR_ROOT / "IsaacLab", VENDOR_ROOT / "IsaacLab-v2.3.2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_isaaclab() -> Path:
    override = os.environ.get("HRVLA_ISAACLAB_ROOT")
    candidates = (Path(override),) if override else ISAACLAB_CANDIDATES
    for candidate in candidates:
        if (candidate / "isaaclab.sh").is_file():
            return candidate
    expected = ", ".join(str(path) for path in candidates)
    raise RuntimeError(f"locked Isaac Lab checkout not found; checked: {expected}")


def verify_inputs(lock: dict[str, object], isaaclab_root: Path) -> None:
    if not RUNTIME_PYTHON.is_file():
        raise RuntimeError(f"SONIC runtime is missing: {RUNTIME_PYTHON}")
    if not (SONIC_ROOT / ".git").exists():
        raise RuntimeError(f"GEAR-SONIC source checkout is missing: {SONIC_ROOT}")

    source_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=SONIC_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    expected_source = str(lock["source"]["revision"])
    if source_revision != expected_source:
        raise RuntimeError(
            f"GEAR-SONIC source is {source_revision}; expected {expected_source}"
        )

    lab_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=isaaclab_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    expected_lab = str(lock["runtime"]["isaac_lab_git_revision"])
    if lab_revision != expected_lab:
        raise RuntimeError(f"Isaac Lab source is {lab_revision}; expected {expected_lab}")

    version_probe = """
import json
import platform
from importlib.metadata import version
names = {
    "isaac_sim": "isaacsim",
    "torch": "torch",
    "warp_lang": "warp-lang",
    "tensordict": "tensordict",
    "numpy": "numpy",
    "hydra_core": "hydra-core",
    "gear_sonic": "gear-sonic",
}
print(json.dumps({"python": platform.python_version(), **{k: version(v) for k, v in names.items()}}))
"""
    actual_runtime = json.loads(
        subprocess.run(
            [str(RUNTIME_PYTHON), "-c", version_probe],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    expected_runtime = lock["runtime"]
    for name, actual_version in actual_runtime.items():
        expected_version = str(expected_runtime[name])
        if actual_version != expected_version:
            raise RuntimeError(
                f"runtime {name} is {actual_version}; expected {expected_version}"
            )

    mismatches = []
    for relative, expected_hash in lock["artifacts"].items():
        artifact = SONIC_ROOT / relative
        if not artifact.is_file():
            mismatches.append(f"missing: {artifact}")
            continue
        actual_hash = sha256(artifact)
        if actual_hash != expected_hash:
            mismatches.append(f"hash mismatch: {artifact} ({actual_hash})")
    if mismatches:
        raise RuntimeError("locked artifact validation failed:\n  " + "\n  ".join(mismatches))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("metrics", "viewer"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "_artifacts" / "sonic_eval" / "default_sample",
        help="Metrics JSON output directory (metrics mode only)",
    )
    parser.add_argument("--num-envs", type=int, help="Override the mode default")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the command")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    isaaclab_root = find_isaaclab()
    verify_inputs(lock, isaaclab_root)

    num_envs = args.num_envs or (2 if args.mode == "metrics" else 1)
    if num_envs < 1:
        raise RuntimeError("--num-envs must be positive")

    command = [
        str(isaaclab_root / "isaaclab.sh"),
        "-p",
        "gear_sonic/eval_agent_trl.py",
        "+checkpoint=sonic_release/last.pt",
        f"+headless={args.mode == 'metrics'}",
        f"++num_envs={num_envs}",
        "++manager_env.observations.policy.enable_corruption=False",
        "++manager_env.observations.tokenizer.enable_corruption=False",
        "++manager_env.commands.motion.motion_lib_cfg.motion_file=sample_data/robot_filtered",
        "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=sample_data/smpl_filtered",
    ]

    if args.mode == "metrics":
        command.extend(
            [
                "++eval_callbacks=im_eval",
                "++run_eval_loop=False",
                "+manager_env/terminations=tracking/eval",
                "++manager_env.commands.motion.motion_lib_cfg.max_unique_motions=2",
                f"+eval_output_dir={args.output_dir.resolve()}",
            ]
        )

    environment = os.environ.copy()
    for variable in ("ISAAC_PATH", "ISAACLAB_PATH", "PYTHONPATH", "LD_LIBRARY_PATH"):
        environment.pop(variable, None)
    environment.update(
        {
            "ACCEPT_EULA": "Y",
            "PATH": f"{RUNTIME_PYTHON.parent}{os.pathsep}{environment['PATH']}",
            "TERM": "xterm-256color",
            "PYTHONUNBUFFERED": "1",
            "VIRTUAL_ENV": str(RUNTIME_PYTHON.parents[1]),
        }
    )

    print("Validated locked SONIC source, runtime, and artifacts.", flush=True)
    print(" ".join(command), flush=True)
    if args.dry_run:
        return 0

    result = subprocess.run(command, cwd=SONIC_ROOT, env=environment, check=False)
    return result.returncode


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
