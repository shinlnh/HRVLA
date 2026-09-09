#!/usr/bin/env python3
"""Validate the baseline source lock without installing the robotics stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = REPO_ROOT / "config" / "upstreams.lock.json"
SHA40 = re.compile(r"^[0-9a-f]{40}$")
OFFICIAL_GIT_PREFIXES = (
    "https://github.com/NVIDIA/",
    "https://github.com/NVlabs/",
    "https://github.com/unitreerobotics/",
    "https://github.com/isaac-sim/",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--vendor-dir", type=Path, default=REPO_ROOT / "_vendor")
    parser.add_argument(
        "--check-local",
        action="store_true",
        help="Require local upstream checkouts and verify their exact HEAD commits",
    )
    return parser.parse_args()


def validate(lock: dict[str, object], vendor: Path, check_local: bool) -> list[str]:
    errors: list[str] = []
    if lock.get("schema_version") != 1:
        errors.append("schema_version must be 1")

    contract = lock.get("stack_contract", {})
    if contract.get("embodiment_tag") != "UNITREE_G1_SONIC":
        errors.append("embodiment_tag must be UNITREE_G1_SONIC")
    if contract.get("isaac_lab") != "2.3.2":
        errors.append("Isaac Lab must remain pinned to the reviewed 2.3.2 baseline")

    git_specs = lock.get("git", {})
    required = {
        "gr00t_whole_body_control",
        "isaac_gr00t",
        "unitree_sim_isaaclab",
        "isaac_lab",
    }
    missing = required - set(git_specs)
    if missing:
        errors.append(f"missing Git component(s): {', '.join(sorted(missing))}")

    for name, spec in git_specs.items():
        revision = str(spec.get("revision", ""))
        url = str(spec.get("url", ""))
        destination = str(spec.get("destination", ""))
        if not SHA40.fullmatch(revision):
            errors.append(f"{name}: revision is not a 40-character Git SHA")
        if not url.startswith(OFFICIAL_GIT_PREFIXES):
            errors.append(f"{name}: non-official Git URL {url!r}")
        if not destination or Path(destination).name != destination:
            errors.append(f"{name}: destination must be one directory name")
        if check_local and destination:
            checkout = vendor / destination
            if not (checkout / ".git").exists():
                errors.append(f"{name}: missing checkout {checkout}")
            else:
                actual = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=checkout,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                if actual != revision:
                    errors.append(f"{name}: local HEAD {actual} != {revision}")

    for name, spec in lock.get("huggingface", {}).items():
        if not SHA40.fullmatch(str(spec.get("revision", ""))):
            errors.append(f"{name}: Hugging Face revision is not a 40-character SHA")
        repo_id = str(spec.get("repo_id", ""))
        if not repo_id.startswith("nvidia/"):
            errors.append(f"{name}: baseline artifact is not from official nvidia namespace")

    return errors


def main() -> int:
    args = parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    errors = validate(lock, args.vendor_dir, args.check_local)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("baseline lock: valid")
    print(f"git components: {len(lock['git'])}")
    print(f"Hugging Face artifacts: {len(lock['huggingface'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
