"""Frozen split and power contracts for the five-row internal benchmark."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping

from .evidence import claim_readiness
from .plan import build_plan, canonical_sha256


INTERNAL_METHODS = (
    "gr00t_sonic",
    "gr00t_st",
    "gr00t_st_rt",
    "gr00t_str",
    "gr00t_str_rt",
)
SPLIT_ORDER = ("development", "validation", "hidden_final")


def _derive_seed(namespace: str, index: int) -> int:
    payload = f"hrvla-internal-v1|{namespace}|{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[-4:], "big") & 0x7FFFFFFF


def mcnemar_exact_two_sided(a_only: int, b_only: int) -> float:
    """Return the same exact conditional McNemar p-value used by the scorer."""

    discordant = a_only + b_only
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, value) for value in range(min(a_only, b_only) + 1))
    return min(1.0, 2 * tail / (2**discordant))


def exact_mcnemar_power(
    pairs: int,
    *,
    p_candidate_only: float,
    p_baseline_only: float,
    alpha: float,
) -> float:
    """Exact multinomial rejection probability for a paired binary alternative."""

    if type(pairs) is not int or pairs < 1:
        raise ValueError("pairs must be a positive integer")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be within (0, 1)")
    if min(p_candidate_only, p_baseline_only) < 0.0:
        raise ValueError("discordant probabilities cannot be negative")
    tied = 1.0 - p_candidate_only - p_baseline_only
    if tied < 0.0:
        raise ValueError("discordant probabilities cannot sum above one")

    power = 0.0
    factorial_pairs = math.factorial(pairs)
    for candidate_only in range(pairs + 1):
        for baseline_only in range(pairs - candidate_only + 1):
            ties = pairs - candidate_only - baseline_only
            probability = (
                factorial_pairs
                / (
                    math.factorial(candidate_only)
                    * math.factorial(baseline_only)
                    * math.factorial(ties)
                )
                * p_candidate_only**candidate_only
                * p_baseline_only**baseline_only
                * tied**ties
            )
            if mcnemar_exact_two_sided(candidate_only, baseline_only) <= alpha:
                power += probability
    return power


def validate_internal_protocol_lock(lock: dict[str, Any]) -> None:
    """Reject a changed split, taxonomy, or underpowered final comparison."""

    if lock.get("schema_version") != 1:
        raise ValueError("internal protocol schema_version must be 1")
    if lock.get("status") != "frozen_before_ours":
        raise ValueError("internal protocol must be frozen before ours")
    if tuple(lock.get("methods", ())) != INTERNAL_METHODS:
        raise ValueError("internal protocol must contain the five registered methods in order")
    if lock.get("training_seeds") != [0, 1, 2]:
        raise ValueError("internal protocol training seeds must remain [0, 1, 2]")
    if lock.get("paired_across_methods") is not True:
        raise ValueError("rollout seeds must be paired across methods")

    splits = lock.get("splits")
    if not isinstance(splits, dict) or tuple(splits) != SPLIT_ORDER:
        raise ValueError("internal protocol split order or membership changed")
    observed: set[int] = set()
    for split_name in SPLIT_ORDER:
        split = splits[split_name]
        count = split.get("rollouts_per_training_seed_per_cell")
        seeds = split.get("rollout_seeds")
        if type(count) is not int or count < 1 or not isinstance(seeds, list):
            raise ValueError(f"{split_name}: invalid rollout count or seeds")
        expected = [_derive_seed(split_name, index) for index in range(count)]
        if seeds != expected:
            raise ValueError(f"{split_name}: rollout seeds differ from the frozen derivation")
        if observed & set(seeds):
            raise ValueError("internal protocol splits overlap")
        observed.update(seeds)
    if splits["development"].get("claim_eligible") is not False:
        raise ValueError("development split cannot be claim eligible")
    if splits["validation"].get("claim_eligible") is not False:
        raise ValueError("validation split cannot be claim eligible")
    if splits["hidden_final"].get("claim_eligible") is not True:
        raise ValueError("hidden_final must be the only claim-eligible split")

    power = lock.get("power_analysis", {})
    families = power.get("primary_comparison_families")
    if not isinstance(families, list) or len(families) != 4 or len(set(families)) != 4:
        raise ValueError("power analysis must predeclare four unique comparison families")
    alpha = float(power.get("familywise_alpha", -1.0))
    corrected_alpha = float(power.get("bonferroni_design_alpha", -1.0))
    if not math.isclose(corrected_alpha, alpha / len(families), abs_tol=1e-15):
        raise ValueError("power design alpha must protect the four-comparison family")
    final_repeats = splits["hidden_final"]["rollouts_per_training_seed_per_cell"]
    pairs = len(lock["training_seeds"]) * final_repeats
    if power.get("paired_observations_per_cell") != pairs:
        raise ValueError("power pair count differs from hidden-final replication")
    alternative = power.get("design_alternative", {})
    calculated = exact_mcnemar_power(
        pairs,
        p_candidate_only=float(alternative.get("candidate_only_success_probability", -1.0)),
        p_baseline_only=float(alternative.get("baseline_only_success_probability", -1.0)),
        alpha=corrected_alpha,
    )
    if not math.isclose(calculated, float(power.get("exact_power", -1.0)), abs_tol=1e-12):
        raise ValueError("reported exact McNemar power differs from the locked design")
    if calculated < float(power.get("minimum_power", 1.0)):
        raise ValueError("hidden-final replication is below the predeclared power target")


def build_internal_plans(
    admitted_suite: dict[str, Any],
    lock: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Create non-overlapping, paired plans only from a fully admitted suite."""

    validate_internal_protocol_lock(lock)
    readiness = claim_readiness(admitted_suite)
    if not readiness["claim_ready"]:
        raise ValueError(
            "internal plans require a fully admitted suite: " + "; ".join(readiness["blockers"])
        )
    source = lock.get("source_suite", {})
    if source.get("suite_id") != admitted_suite.get("suite_id"):
        raise ValueError("internal protocol source suite ID differs")
    expected_draft_hash = source.get("draft_suite_sha256")
    provenance = admitted_suite.get("admission_provenance", {})
    if provenance.get("draft_suite_sha256") != expected_draft_hash:
        raise ValueError("admitted suite does not descend from the frozen draft suite")

    plans = {}
    for split_name in SPLIT_ORDER:
        plans[split_name] = build_plan(
            admitted_suite,
            INTERNAL_METHODS,
            rollout_seeds=lock["splits"][split_name]["rollout_seeds"],
        )
    episode_keys = {
        split_name: {episode["episode_key"] for episode in plan["episodes"]}
        for split_name, plan in plans.items()
    }
    for index, split_a in enumerate(SPLIT_ORDER):
        for split_b in SPLIT_ORDER[index + 1 :]:
            if episode_keys[split_a] & episode_keys[split_b]:
                raise AssertionError(f"internal split overlap: {split_a}/{split_b}")
    return plans


def protocol_summary(
    plans: Mapping[str, dict[str, Any]], lock: dict[str, Any]
) -> dict[str, Any]:
    """Return an auditable experiment-count summary without method outcomes."""

    validate_internal_protocol_lock(lock)
    core = {
        "schema_version": 1,
        "methods": list(INTERNAL_METHODS),
        "training_seeds": lock["training_seeds"],
        "splits": {
            name: {
                "plan_sha256": plans[name]["plan_sha256"],
                "episodes_per_method": len(plans[name]["episodes"]),
                "method_episode_records": len(plans[name]["episodes"]) * len(INTERNAL_METHODS),
                "claim_eligible": lock["splits"][name]["claim_eligible"],
            }
            for name in SPLIT_ORDER
        },
        "power_analysis": lock["power_analysis"],
    }
    return {**core, "summary_sha256": canonical_sha256(core)}


__all__ = [
    "INTERNAL_METHODS",
    "SPLIT_ORDER",
    "build_internal_plans",
    "exact_mcnemar_power",
    "mcnemar_exact_two_sided",
    "protocol_summary",
    "validate_internal_protocol_lock",
]
