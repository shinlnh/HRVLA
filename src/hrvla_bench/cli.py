"""Command-line interface for planning, validating, and ingesting evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .evidence import audit_evidence, claim_readiness
from .execution import RunIdentity, load_adapter, run_plan
from .humanoidarena import admission_report
from .methods import validate_method_registry
from .plan import build_plan, load_json, validate_suite, write_plan
from .provenance import collect_provenance, write_manifest
from .score import load_jsonl, score_records, validate_record
from .sonic import disturbance_pilot_report, sonic_records, write_jsonl


REPO_ROOT = Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-suite", help="Validate suite invariants")
    validate.add_argument("suite", type=Path)

    methods = sub.add_parser("validate-methods", help="Validate ST/STR/RT baselines")
    methods.add_argument("registry", type=Path)
    methods.add_argument("--artifact-lock", type=Path, required=True)

    humanoid = sub.add_parser(
        "check-humanoidarena", help="Check locked HumanoidArena admission inputs"
    )
    humanoid.add_argument("lock", type=Path)
    humanoid.add_argument("--source-root", type=Path)
    humanoid.add_argument("--model-root", type=Path)
    humanoid.add_argument("--dataset-root", type=Path)
    humanoid.add_argument("--asset-root", type=Path)
    humanoid.add_argument("--sonic-policy-root", type=Path)
    humanoid.add_argument("--oracle-evidence", type=Path)
    humanoid.add_argument("--isaac-sim-version")
    humanoid.add_argument("--isaac-lab-ref")
    humanoid.add_argument("--output", type=Path)

    readiness = sub.add_parser(
        "claim-readiness", help="Audit whether a suite can produce claim-bearing evidence"
    )
    readiness.add_argument("suite", type=Path)
    readiness.add_argument("--output", type=Path)

    plan = sub.add_parser("plan", help="Generate a deterministic paired episode plan")
    plan.add_argument("suite", type=Path)
    plan.add_argument("--method", action="append", required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--include-draft", action="store_true")
    plan.add_argument("--rollouts", type=int)
    plan.add_argument("--manifest", type=Path)

    records = sub.add_parser("validate-records", help="Validate episode JSONL")
    records.add_argument("episodes", type=Path)

    ingest = sub.add_parser("ingest-sonic", help="Ingest official SONIC motion metrics")
    ingest.add_argument("metrics", type=Path)
    ingest.add_argument("--output", type=Path, required=True)
    ingest.add_argument("--run-id", required=True)
    ingest.add_argument("--checkpoint-id", required=True)
    ingest.add_argument("--controller-id", required=True)
    ingest.add_argument("--simulator-revision", required=True)
    ingest.add_argument("--recorded-at")

    disturbance = sub.add_parser(
        "report-sonic-disturbance",
        help="Report an audited, non-claim-bearing SONIC body-push pilot",
    )
    disturbance.add_argument("nominal_metrics", type=Path)
    disturbance.add_argument("disturbed_metrics", type=Path)
    disturbance.add_argument("audit", type=Path)
    disturbance.add_argument("--output", type=Path, required=True)
    disturbance.add_argument("--run-id", required=True)
    disturbance.add_argument("--checkpoint-id", required=True)
    disturbance.add_argument("--controller-id", required=True)
    disturbance.add_argument("--simulator-revision", required=True)
    disturbance.add_argument("--recorded-at")

    run = sub.add_parser("run", help="Execute one method through a Python adapter")
    run.add_argument("plan", type=Path)
    run.add_argument("--adapter", required=True)
    run.add_argument("--method", required=True)
    run.add_argument("--track", required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--checkpoint-id", required=True)
    run.add_argument("--controller-id", required=True)
    run.add_argument("--simulator-revision", required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--allow-draft", action="store_true")

    score = sub.add_parser("score", help="Score validated episode JSONL")
    score.add_argument("episodes", type=Path)
    score.add_argument("--output", type=Path)

    evidence = sub.add_parser(
        "audit-evidence", help="Verify complete matched evidence for every planned method"
    )
    evidence.add_argument("plan", type=Path)
    evidence.add_argument("episodes", type=Path, nargs="+")
    evidence.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate-suite":
            validate_suite(load_json(args.suite))
            print("suite: valid")
        elif args.command == "validate-methods":
            validate_method_registry(
                load_json(args.registry), load_json(args.artifact_lock), REPO_ROOT
            )
            print("method registry: valid")
        elif args.command == "check-humanoidarena":
            report = admission_report(
                load_json(args.lock),
                REPO_ROOT,
                source_root=args.source_root,
                model_root=args.model_root,
                dataset_root=args.dataset_root,
                asset_root=args.asset_root,
                sonic_policy_root=args.sonic_policy_root,
                oracle_evidence=args.oracle_evidence,
                isaac_sim_version=args.isaac_sim_version,
                isaac_lab_ref=args.isaac_lab_ref,
            )
            rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(rendered, encoding="utf-8")
            else:
                print(rendered, end="")
            if not report["admitted"]:
                return 3
        elif args.command == "claim-readiness":
            report = claim_readiness(load_json(args.suite))
            rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(rendered, encoding="utf-8")
            else:
                print(rendered, end="")
            if not report["claim_ready"]:
                return 3
        elif args.command == "plan":
            plan = build_plan(
                load_json(args.suite),
                args.method,
                include_draft=args.include_draft,
                rollouts_override=args.rollouts,
            )
            write_plan(plan, args.output)
            if args.manifest:
                write_manifest(collect_provenance(REPO_ROOT, plan), args.manifest)
            print(f"episodes: {len(plan['episodes'])}")
            print(f"plan_sha256: {plan['plan_sha256']}")
        elif args.command == "validate-records":
            loaded = load_jsonl(args.episodes)
            for record in loaded:
                validate_record(record)
            print(f"episode records: {len(loaded)} valid")
        elif args.command == "ingest-sonic":
            records = sonic_records(
                args.metrics,
                checkpoint_id=args.checkpoint_id,
                controller_id=args.controller_id,
                simulator_revision=args.simulator_revision,
                run_id=args.run_id,
                recorded_at=args.recorded_at,
            )
            for record in records:
                validate_record(record)
            write_jsonl(records, args.output)
            print(f"episode records: {len(records)}")
        elif args.command == "report-sonic-disturbance":
            report = disturbance_pilot_report(
                args.nominal_metrics,
                args.disturbed_metrics,
                args.audit,
                checkpoint_id=args.checkpoint_id,
                controller_id=args.controller_id,
                simulator_revision=args.simulator_revision,
                run_id=args.run_id,
                recorded_at=args.recorded_at,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(f"disturbance pilot: {args.output}")
        elif args.command == "run":
            identity = RunIdentity(
                run_id=args.run_id,
                method_id=args.method,
                evaluation_track=args.track,
                policy_checkpoint_id=args.checkpoint_id,
                controller_id=args.controller_id,
                simulator_revision=args.simulator_revision,
            )
            records = run_plan(
                load_json(args.plan),
                identity,
                load_adapter(args.adapter),
                args.output,
                allow_draft=args.allow_draft,
            )
            print(f"episode records: {len(records)}")
        elif args.command == "score":
            report = score_records(load_jsonl(args.episodes))
            rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(rendered, encoding="utf-8")
            else:
                print(rendered, end="")
        elif args.command == "audit-evidence":
            records = []
            for path in args.episodes:
                records.extend(load_jsonl(path))
            report = audit_evidence(load_json(args.plan), records)
            rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(rendered, encoding="utf-8")
            else:
                print(rendered, end="")
    except (KeyError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
