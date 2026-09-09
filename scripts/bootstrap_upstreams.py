#!/usr/bin/env python3
"""Clone official upstream sources and detach them at locked revisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = REPO_ROOT / "config" / "upstreams.lock.json"
DEFAULT_VENDOR = REPO_ROOT / "_vendor"


def run(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        args,
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.stdout.strip()


def normalized_remote(url: str) -> str:
    return url.removesuffix("/").removesuffix(".git")


def checkout_component(name: str, spec: dict[str, object], vendor: Path, pull_lfs: bool) -> None:
    url = str(spec["url"])
    revision = str(spec["revision"])
    destination = vendor / str(spec["destination"])

    if destination.exists():
        if not (destination / ".git").exists():
            raise RuntimeError(f"{destination} exists but is not a Git checkout")
        actual_remote = run("git", "remote", "get-url", "origin", cwd=destination)
        if normalized_remote(actual_remote) != normalized_remote(url):
            raise RuntimeError(
                f"{destination} origin is {actual_remote!r}; expected {url!r}"
            )
    else:
        print(f"[{name}] cloning {url}")
        run(
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            url,
            str(destination),
        )

    current = (
        run("git", "rev-parse", "HEAD", cwd=destination)
        if (destination / ".git" / "HEAD").exists()
        else ""
    )
    if current != revision:
        print(f"[{name}] fetching {revision}")
        run("git", "fetch", "--depth", "1", "origin", revision, cwd=destination)
        run("git", "checkout", "--detach", revision, cwd=destination)

    actual = run("git", "rev-parse", "HEAD", cwd=destination)
    if actual != revision:
        raise RuntimeError(f"{name}: checkout is {actual}, expected {revision}")

    if pull_lfs:
        run("git", "lfs", "install", "--local", cwd=destination)
        run("git", "lfs", "pull", cwd=destination)

    print(f"[{name}] ready at {actual}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--vendor-dir", type=Path, default=DEFAULT_VENDOR)
    parser.add_argument(
        "--component",
        action="append",
        help="Bootstrap only this lock-file component; repeat for multiple components",
    )
    parser.add_argument(
        "--lfs",
        action="store_true",
        help="Also pull Git LFS objects; omitted by default to avoid large downloads",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if shutil.which("git") is None:
        raise RuntimeError("git is required")
    if args.lfs and shutil.which("git-lfs") is None:
        raise RuntimeError("git-lfs is required with --lfs")

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    components = lock["git"]
    selected = args.component or list(components)
    unknown = sorted(set(selected) - set(components))
    if unknown:
        raise RuntimeError(f"unknown component(s): {', '.join(unknown)}")

    args.vendor_dir.mkdir(parents=True, exist_ok=True)
    for name in selected:
        checkout_component(name, components[name], args.vendor_dir, args.lfs)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
