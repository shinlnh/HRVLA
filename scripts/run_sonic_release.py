#!/usr/bin/env python3
"""Run the locked NVIDIA SONIC release in its isolated Isaac Lab runtime."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
PIP_RUNTIME_PYTHON = VENDOR_ROOT / "sonic-runtime" / "bin" / "python"
ISAACLAB_CANDIDATES = (VENDOR_ROOT / "IsaacLab", VENDOR_ROOT / "IsaacLab-v2.3.2")


@dataclass(frozen=True)
class Runtime:
    """Resolved simulator Python launcher and its environment contract."""

    kind: str
    python: Path
    isaacsim_root: Path | None = None


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


def resolve_runtime(kind: str, isaaclab_root: Path) -> Runtime:
    workstation_root = Path(
        os.environ.get("HRVLA_ISAACSIM_ROOT", str(isaaclab_root / "_isaac_sim"))
    ).resolve()
    workstation_python = workstation_root / "python.sh"

    if kind in {"auto", "workstation"} and workstation_python.is_file():
        return Runtime("workstation", workstation_python, workstation_root)
    if kind == "workstation":
        raise RuntimeError(
            "Isaac Sim workstation runtime is missing; expected "
            f"{workstation_python} or set HRVLA_ISAACSIM_ROOT"
        )
    if kind in {"auto", "pip"} and PIP_RUNTIME_PYTHON.is_file():
        return Runtime("pip", PIP_RUNTIME_PYTHON)
    raise RuntimeError(f"SONIC pip runtime is missing: {PIP_RUNTIME_PYTHON}")


def verify_inputs(
    lock: dict[str, object], isaaclab_root: Path, runtime: Runtime
) -> None:
    if not runtime.python.is_file():
        raise RuntimeError(f"SONIC runtime is missing: {runtime.python}")
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
    runtime_lock = lock["runtime"]
    expected_lab = str(runtime_lock["isaac_lab_git_revision"])
    if lab_revision != expected_lab:
        raise RuntimeError(f"Isaac Lab source is {lab_revision}; expected {expected_lab}")

    version_probe = """
import json
import platform
from importlib.metadata import version
names = {
    "torch": "torch",
    "warp_lang": "warp-lang",
    "tensordict": "tensordict",
    "numpy": "numpy",
    "hydra_core": "hydra-core",
    "gear_sonic": "gear-sonic",
}
print(json.dumps({"python": platform.python_version(), **{k: version(v) for k, v in names.items()}}))
"""
    version_output = subprocess.run(
        [str(runtime.python), "-c", version_probe],
        check=True,
        capture_output=True,
        text=True,
    )
    output_lines = [line for line in version_output.stdout.splitlines() if line.strip()]
    if not output_lines:
        raise RuntimeError(f"runtime version probe returned no output: {runtime.python}")
    actual_runtime = json.loads(output_lines[-1])
    expected_runtime = {
        **runtime_lock["common"],
        **runtime_lock["profiles"][runtime.kind],
    }
    for name, actual_version in actual_runtime.items():
        expected_version = str(expected_runtime[name])
        if actual_version != expected_version:
            raise RuntimeError(
                f"runtime {name} is {actual_version}; expected {expected_version}"
            )

    if runtime.kind == "pip":
        isaac_sim_version = subprocess.run(
            [
                str(runtime.python),
                "-c",
                "from importlib.metadata import version; print(version('isaacsim'))",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().splitlines()[-1]
    else:
        version_file = runtime.isaacsim_root / "VERSION"
        if not version_file.is_file():
            raise RuntimeError(f"Isaac Sim VERSION file is missing: {version_file}")
        isaac_sim_version = version_file.read_text(encoding="utf-8").strip().split("-")[0]
    expected_isaac_sim = str(expected_runtime["isaac_sim"])
    if isaac_sim_version != expected_isaac_sim:
        raise RuntimeError(
            f"runtime isaac_sim is {isaac_sim_version}; expected {expected_isaac_sim}"
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
    parser.add_argument("mode", choices=("metrics", "record", "viewer"))
    parser.add_argument(
        "--runtime",
        choices=("auto", "pip", "workstation"),
        default="auto",
        help="Simulator installation to use; auto prefers the workstation binary",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "_artifacts" / "sonic_eval" / "default_sample",
        help="Metrics or recording output directory",
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        help="Completed metrics_eval.json used to select motions in record mode",
    )
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--render-height", type=int, default=540)
    parser.add_argument(
        "--inject-push",
        action="store_true",
        help="retain SONIC's configured interval velocity-push event during evaluation",
    )
    parser.add_argument("--num-envs", type=int, help="Override the mode default")
    parser.add_argument(
        "--viewer-eye",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(4.5, 0.0, 4.0),
        help="Viewer camera position in metres (viewer mode only)",
    )
    parser.add_argument(
        "--viewer-lookat",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
        help="Viewer camera target in metres (viewer mode only)",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="+HYDRA.KEY=VALUE",
        help="Append an explicit Hydra override; repeat for benchmark perturbations",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the command")
    return parser.parse_args()


def hydra_vector(values: tuple[float, float, float] | list[float]) -> str:
    """Serialize a three-dimensional vector without shell quoting."""
    return "[" + ",".join(format(value, "g") for value in values) + "]"


def main() -> int:
    args = parse_args()
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    isaaclab_root = find_isaaclab()
    runtime = resolve_runtime(args.runtime, isaaclab_root)
    verify_inputs(lock, isaaclab_root, runtime)

    num_envs = args.num_envs or (2 if args.mode == "metrics" else 1)
    if num_envs < 1:
        raise RuntimeError("--num-envs must be positive")
    if args.mode == "record" and (
        args.metrics_file is None or not args.metrics_file.is_file()
    ):
        raise RuntimeError("record mode requires an existing --metrics-file")
    if args.render_width < 1 or args.render_height < 1:
        raise RuntimeError("render dimensions must be positive")
    if any(not override.startswith("+") for override in args.override):
        raise RuntimeError("benchmark overrides must start with '+'")

    command = [
        str(isaaclab_root / "isaaclab.sh"),
        "-p",
        "gear_sonic/eval_agent_trl.py",
        "+checkpoint=sonic_release/last.pt",
        f"+headless={args.mode != 'viewer'}",
        f"++num_envs={num_envs}",
        "++manager_env.observations.policy.enable_corruption=False",
        "++manager_env.observations.tokenizer.enable_corruption=False",
        "++manager_env.commands.motion.motion_lib_cfg.motion_file=sample_data/robot_filtered",
        "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=sample_data/smpl_filtered",
    ]
    if args.inject_push:
        # The release config classifies push_robot as train-only. Clearing that
        # removal list retains its deterministic seeded interval event for a
        # disturbance-recovery regression without editing vendored source.
        command.append("++manager_env.config.train_only_events=[]")

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
    elif args.mode == "record":
        recording_dir = args.output_dir.resolve() / "render_results"
        command.extend(
            [
                "++eval_callbacks=im_eval",
                "++run_eval_loop=False",
                f"++metrics_file={args.metrics_file.resolve()}",
                f"++manager_env.config.save_rendering_dir={recording_dir}",
                "++manager_env.config.render_results=True",
                "++manager_env.config.env_spacing=10.0",
                f"++manager_env.config.render_width={args.render_width}",
                f"++manager_env.config.render_height={args.render_height}",
                "++manager_env.config.render_frame_skip=4",
                "manager_env/recorders=render",
            ]
        )
    else:
        command.extend(
            [
                "++manager_env.config.viewer.eye=" + hydra_vector(args.viewer_eye),
                "++manager_env.config.viewer.lookat="
                + hydra_vector(args.viewer_lookat),
            ]
        )
    command.extend(args.override)

    environment = os.environ.copy()
    for variable in (
        "ISAAC_PATH",
        "ISAACLAB_PATH",
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
        "VIRTUAL_ENV",
    ):
        environment.pop(variable, None)
    environment.update(
        {"ACCEPT_EULA": "Y", "TERM": "xterm-256color", "PYTHONUNBUFFERED": "1"}
    )
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    if runtime.kind == "pip":
        environment["PATH"] = f"{runtime.python.parent}{os.pathsep}{environment['PATH']}"
        environment["VIRTUAL_ENV"] = str(runtime.python.parents[1])
    else:
        environment["ISAAC_PATH"] = str(runtime.isaacsim_root)

    print(
        f"Validated locked SONIC source, {runtime.kind} runtime, and artifacts.",
        flush=True,
    )
    print(" ".join(command), flush=True)
    if args.dry_run:
        return 0

    result = subprocess.run(command, cwd=SONIC_ROOT, env=environment, check=False)
    return result.returncode


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (
        json.JSONDecodeError,
        KeyError,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
