"""Source-observed language-program runtime for the five internal HA rows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from hrvla_subtask.backends import HeuristicProposalBackend
from hrvla_subtask.model import ExecutionMemory, TaskSpec
from hrvla_subtask.planner import PlannerConfig, WorldModelGuidedPlanner

from .humanoidarena_recovery_signals import signals_for_detector
from .methods import EXPECTED_FEATURES
from .recovery_event_detector import SemanticEventDetector


def load_method_programs(path: Path) -> dict[str, Any]:
    programs = json.loads(path.read_text(encoding="utf-8"))
    validate_method_programs(programs)
    return programs


def validate_method_programs(programs: dict[str, Any]) -> None:
    if programs.get("schema_version") != 1:
        raise ValueError("method programs schema_version must be 1")
    planner = programs.get("planner", {})
    expected_planner = {
        "method": "adaptive_ttc",
        "branching_factor": 4,
        "beam_width": 3,
        "search_depth": 3,
        "confidence_threshold": 0.72,
        "proposal_backend": "deterministic-affordance-proposal",
    }
    for key, expected in expected_planner.items():
        if planner.get(key) != expected:
            raise ValueError(f"method planner {key} differs from the frozen contract")
    tasks = programs.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 7:
        raise ValueError("method programs require exactly seven tasks")
    task_ids = [task.get("task_id") for task in tasks]
    if len(set(task_ids)) != 7:
        raise ValueError("method program task IDs must be unique")
    for task in tasks:
        skills = task.get("skills")
        transitions = task.get("transitions")
        if not isinstance(skills, list) or len(skills) < 2:
            raise ValueError(f"{task.get('task_id')}: at least two bounded skills are required")
        if not isinstance(transitions, list) or len(transitions) != len(skills) - 1:
            raise ValueError(f"{task['task_id']}: every non-final skill needs one transition")
        skill_ids = [skill.get("id") for skill in skills]
        if len(skill_ids) != len(set(skill_ids)):
            raise ValueError(f"{task['task_id']}: skill IDs must be unique")
        if [row.get("completes_skill") for row in transitions] != skill_ids[:-1]:
            raise ValueError(f"{task['task_id']}: transition order differs from skill order")
        raw_task = _task_spec_payload(task)
        TaskSpec.from_dict(raw_task)
    recovery = programs.get("recovery_instructions")
    if not isinstance(recovery, dict) or len(recovery) != 9:
        raise ValueError("method programs require exactly nine recovery instructions")
    if any(not isinstance(value, str) or not value.strip() for value in recovery.values()):
        raise ValueError("recovery instructions must be non-empty strings")


def _task_spec_payload(program: dict[str, Any]) -> dict[str, Any]:
    skills = []
    for index, skill in enumerate(program["skills"]):
        row = dict(skill)
        row.setdefault("cost", 1.0)
        row.setdefault("risk", 0.0)
        row.setdefault("success_probability", 1.0)
        row["required"] = True
        skills.append(row)
    return {
        "task_id": program["task_id"],
        "goal_instruction": program["goal_instruction"],
        "initial_state": [],
        "goal_state": list(skills[-1]["adds"]),
        "skills": skills,
        "max_steps": len(skills) + 4,
    }


@dataclass
class MethodInstruction:
    instruction: str
    route: str
    selected_skill_id: str | None
    transition_triggered: bool
    planner_decision: dict[str, Any] | None
    transition_diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HumanoidArenaMethodRuntime:
    """Choose only the language instruction; GR00T/SONIC action contracts stay unchanged."""

    def __init__(
        self,
        programs: dict[str, Any],
        *,
        method_id: str,
        task_id: str,
        control_dt_s: float,
        seed: int,
    ) -> None:
        validate_method_programs(programs)
        if method_id not in EXPECTED_FEATURES:
            raise ValueError(f"unknown internal method: {method_id}")
        program = next((row for row in programs["tasks"] if row["task_id"] == task_id), None)
        if program is None:
            raise ValueError(f"unknown method-program task: {task_id}")
        self.programs = programs
        self.features = EXPECTED_FEATURES[method_id]
        self.method_id = method_id
        self.task_id = task_id
        self.program = program
        self.task = TaskSpec.from_dict(_task_spec_payload(program))
        config = programs["planner"]
        self.backend = HeuristicProposalBackend(error_rate=0.0)
        self.planner = WorldModelGuidedPlanner(
            self.backend,
            PlannerConfig(
                method=config["method"],
                branching_factor=int(config["branching_factor"]),
                beam_width=int(config["beam_width"]),
                search_depth=int(config["search_depth"]),
                confidence_threshold=float(config["confidence_threshold"]),
            ),
        )
        self.memory = ExecutionMemory()
        self.seed = int(seed)
        self.state: frozenset[str] = frozenset()
        self.detectors = [
            SemanticEventDetector(row["detector"], control_dt_s=control_dt_s)
            for row in program["transitions"]
        ]
        self.transition_index = 0
        self.decisions = 0
        self.recovery_decisions = 0

    def reset(self, env: Any) -> None:
        self.memory = ExecutionMemory()
        self.state = frozenset()
        self.transition_index = 0
        self.decisions = 0
        self.recovery_decisions = 0
        for detector in self.detectors:
            detector.reset(env)

    def _observe_transition(
        self, env: Any, *, task_success: bool, semantic_action: Any
    ) -> tuple[bool, dict[str, Any]]:
        if self.transition_index >= len(self.detectors):
            return False, {}
        transition = self.program["transitions"][self.transition_index]
        detector = self.detectors[self.transition_index]
        if detector.detector_id == "first-hand-close" and semantic_action is None:
            return False, {"waiting_for_semantic_action40": True}
        signals = signals_for_detector(detector.detector_id, env, detector.parameters)
        observation = detector.observe(
            env,
            task_success=task_success,
            semantic_action=semantic_action,
            signals=signals,
        )
        if observation.triggered:
            skill = self.task.skill_map[transition["completes_skill"]]
            self.state = skill.apply(self.state)
            self.transition_index += 1
        return observation.triggered, observation.diagnostics

    def instruction(
        self,
        env: Any,
        *,
        task_success: bool = False,
        semantic_action: Any = None,
        recovery_scenario_id: str | None = None,
        recovery_active: bool = False,
    ) -> MethodInstruction:
        if self.features["recovery"] and recovery_active:
            if recovery_scenario_id not in self.programs["recovery_instructions"]:
                raise ValueError("active recovery has no frozen bounded instruction")
            self.recovery_decisions += 1
            return MethodInstruction(
                instruction=self.programs["recovery_instructions"][recovery_scenario_id],
                route="transition_recovery",
                selected_skill_id=None,
                transition_triggered=False,
                planner_decision=None,
                transition_diagnostics={},
            )
        if not self.features["subtask"]:
            self.decisions += 1
            return MethodInstruction(
                instruction=self.program["goal_instruction"],
                route="whole_task",
                selected_skill_id=None,
                transition_triggered=False,
                planner_decision=None,
                transition_diagnostics={},
            )

        triggered, diagnostics = self._observe_transition(
            env, task_success=task_success, semantic_action=semantic_action
        )
        decision = self.planner.decide(
            self.task,
            self.state,
            self.memory,
            seed=self.seed + self.decisions,
        )
        if decision.selected is None:
            instruction = self.program["goal_instruction"]
            skill_id = None
        else:
            skill_id = decision.selected.skill_id
            instruction = self.task.skill_map[skill_id].instruction
        self.decisions += 1
        return MethodInstruction(
            instruction=instruction,
            route=decision.route,
            selected_skill_id=skill_id,
            transition_triggered=triggered,
            planner_decision=decision.to_dict(),
            transition_diagnostics=diagnostics,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "program_id": self.programs["program_id"],
            "method_id": self.method_id,
            "task_id": self.task_id,
            "features": self.features,
            "planner_decisions": self.decisions,
            "recovery_decisions": self.recovery_decisions,
            "completed_transition_count": self.transition_index,
            "total_transition_count": len(self.detectors),
            "memory": self.memory.to_dict(),
        }


__all__ = [
    "HumanoidArenaMethodRuntime",
    "MethodInstruction",
    "load_method_programs",
    "validate_method_programs",
]
