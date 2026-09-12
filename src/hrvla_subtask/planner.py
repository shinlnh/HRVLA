"""Adaptive world-model-guided beam search over language subtasks."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

from .backends import ProposalBackend
from .model import Candidate, Decision, ExecutionMemory, SearchBranch, Skill, TaskSpec


@dataclass(frozen=True)
class PlannerConfig:
    method: str = "adaptive_ttc"
    branching_factor: int = 4
    beam_width: int = 3
    search_depth: int = 3
    confidence_threshold: float = 0.72
    fallback_to_affordance: bool = True

    def validate(self) -> None:
        valid = {"plan_once", "recursive", "best_of_n", "ttc", "adaptive_ttc"}
        if self.method not in valid:
            raise ValueError(f"method must be one of {sorted(valid)}")
        if min(self.branching_factor, self.beam_width, self.search_depth) < 1:
            raise ValueError("search dimensions must be positive")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be within [0, 1]")


class SymbolicWorldModel:
    """Auditable outcome model for skills with declared preconditions and effects."""

    def predict(self, task: TaskSpec, state: frozenset[str], skill_id: str) -> frozenset[str]:
        skill = task.skill_map.get(skill_id)
        if skill is None or not skill.applicable(state):
            return state
        return skill.apply(state)


class PrerequisiteValueModel:
    """Scores predicted states without access to the benchmark's next-action label."""

    def distance_to_goal(self, task: TaskSpec, start: frozenset[str]) -> int:
        if task.complete(start):
            return 0
        frontier = {start}
        visited = {start}
        limit = min(task.max_steps, len(task.skills) + 4)
        for depth in range(1, limit + 1):
            following: set[frozenset[str]] = set()
            for state in frontier:
                for skill in task.skills:
                    if skill.applicable(state) and skill.useful(state):
                        predicted = skill.apply(state)
                        if task.complete(predicted):
                            return depth
                        if predicted not in visited:
                            visited.add(predicted)
                            following.add(predicted)
            if not following:
                break
            frontier = following
        return limit + 2

    def score(
        self,
        task: TaskSpec,
        before: frozenset[str],
        after: frozenset[str],
        skill: Skill | None,
    ) -> float:
        if skill is None:
            return -12.0
        if before == after:
            return -8.0 - skill.risk
        old_distance = self.distance_to_goal(task, before)
        new_distance = self.distance_to_goal(task, after)
        goal_gain = len(task.goal_state & after) - len(task.goal_state & before)
        milestone_gain = int(skill.milestone not in before and skill.milestone in after)
        return (
            3.5 * (old_distance - new_distance)
            + 2.0 * goal_gain
            + 0.75 * milestone_gain
            - 0.08 * skill.cost
            - 0.7 * skill.risk
        )


class WorldModelGuidedPlanner:
    """τ₀-style selective TTC while preserving GR00T's language interface.

    This is an independently implemented algorithmic reproduction.  It does not contain or
    claim the unreleased τ₀-VLA high-level checkpoint.
    """

    def __init__(
        self,
        backend: ProposalBackend,
        config: PlannerConfig | None = None,
        *,
        world_model: SymbolicWorldModel | None = None,
        value_model: PrerequisiteValueModel | None = None,
    ) -> None:
        self.backend = backend
        self.config = config or PlannerConfig()
        self.config.validate()
        self.world_model = world_model or SymbolicWorldModel()
        self.value_model = value_model or PrerequisiteValueModel()

    def _valid(self, task: TaskSpec, state: frozenset[str], candidate: Candidate) -> bool:
        skill = task.skill_map.get(candidate.skill_id)
        return bool(skill and skill.applicable(state) and skill.useful(state))

    def _shield(
        self, task: TaskSpec, state: frozenset[str], candidate: Candidate | None
    ) -> tuple[Candidate | None, bool]:
        if candidate is not None and self._valid(task, state, candidate):
            return candidate, False
        if not self.config.fallback_to_affordance:
            return candidate, False
        scored: list[tuple[float, Skill]] = []
        for skill in task.skills:
            if not skill.applicable(state) or not skill.useful(state):
                continue
            predicted = self.world_model.predict(task, state, skill.skill_id)
            scored.append((self.value_model.score(task, state, predicted, skill), skill))
        if not scored:
            return None, candidate is not None
        _, skill = max(scored, key=lambda item: (item[0], -item[1].priority, item[1].skill_id))
        return Candidate(skill.skill_id, 0.0, "affordance safety fallback"), True

    def _search(
        self,
        task: TaskSpec,
        state: frozenset[str],
        memory: ExecutionMemory,
        seed: int,
        depth: int,
    ) -> tuple[Candidate | None, tuple[Candidate, ...], tuple[str, ...], int]:
        beam = [SearchBranch(state=state, path=(), score=0.0, memory=memory.clone())]
        all_candidates: list[Candidate] = []
        nodes = 0
        for level in range(depth):
            expanded: list[SearchBranch] = []
            for branch_index, branch in enumerate(beam):
                proposals = self.backend.propose(
                    task,
                    branch.state,
                    branch.memory,
                    self.config.branching_factor,
                    seed + level * 1009 + branch_index * 97,
                )
                all_candidates.extend(proposals)
                for candidate in proposals:
                    nodes += 1
                    skill = task.skill_map.get(candidate.skill_id)
                    predicted = self.world_model.predict(task, branch.state, candidate.skill_id)
                    local = self.value_model.score(task, branch.state, predicted, skill)
                    confidence_bonus = math.log(max(candidate.confidence, 1e-6)) * 0.15
                    branch_memory = branch.memory.clone()
                    if skill and predicted != branch.state:
                        branch_memory.completed.append(skill.skill_id)
                        branch_memory.last_subtask = skill.skill_id
                    expanded.append(
                        SearchBranch(
                            state=predicted,
                            path=branch.path + (candidate.skill_id,),
                            score=branch.score + local + confidence_bonus,
                            memory=branch_memory,
                        )
                    )
            if not expanded:
                break
            expanded.sort(key=lambda item: (item.score, tuple(reversed(item.path))), reverse=True)
            beam = expanded[: self.config.beam_width]
            if all(task.complete(branch.state) for branch in beam):
                break
        if not beam or not beam[0].path:
            return None, tuple(all_candidates), (), nodes
        selected_id = beam[0].path[0]
        selected = next(
            (candidate for candidate in all_candidates if candidate.skill_id == selected_id),
            Candidate(selected_id, 0.0, "beam-reflected selection"),
        )
        return selected, tuple(all_candidates), beam[0].path, nodes

    def _recursive_rollout(
        self,
        task: TaskSpec,
        state: frozenset[str],
        memory: ExecutionMemory,
        direct: Candidate,
        seed: int,
    ) -> tuple[tuple[Candidate, ...], tuple[str, ...], int]:
        """Anticipation/STEP-style recursive subgoal rollout without alternative search."""
        candidates = [direct]
        path = [direct.skill_id]
        predicted = self.world_model.predict(task, state, direct.skill_id)
        branch_memory = memory.clone()
        for level in range(1, self.config.search_depth):
            if task.complete(predicted):
                break
            proposal = self.backend.propose(
                task, predicted, branch_memory, 1, seed + level * 1009
            )[0]
            candidates.append(proposal)
            path.append(proposal.skill_id)
            predicted = self.world_model.predict(task, predicted, proposal.skill_id)
            branch_memory.last_subtask = proposal.skill_id
        return tuple(candidates), tuple(path), len(candidates)

    def decide(
        self,
        task: TaskSpec,
        state: frozenset[str],
        memory: ExecutionMemory,
        *,
        seed: int = 0,
    ) -> Decision:
        started = time.perf_counter()
        before_calls = self.backend.calls
        memory_update = memory.reconcile(task, state)
        if task.complete(state):
            return Decision(
                selected=None,
                route="done",
                confidence=1.0,
                memory_update=memory_update,
                candidates=(),
                search_nodes=0,
                model_calls=0,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        direct = self.backend.propose(task, state, memory, 1, seed)[0]
        candidates: tuple[Candidate, ...] = (direct,)
        selected: Candidate | None = direct
        path: tuple[str, ...] = (direct.skill_id,)
        nodes = 1
        route = "plan_once"

        should_search = self.config.method == "ttc"
        if self.config.method == "ttc":
            route = "ttc"
        elif self.config.method == "adaptive_ttc":
            should_search = direct.confidence < self.config.confidence_threshold or not self._valid(
                task, state, direct
            )
            route = "ttc" if should_search else "fast"
        elif self.config.method == "best_of_n":
            route = "best_of_n"
            proposals = self.backend.propose(
                task, state, memory, self.config.branching_factor, seed + 17
            )
            candidates = tuple(proposals)
            nodes = len(proposals)
            ranked = []
            for proposal in proposals:
                skill = task.skill_map.get(proposal.skill_id)
                predicted = self.world_model.predict(task, state, proposal.skill_id)
                ranked.append((self.value_model.score(task, state, predicted, skill), proposal))
            selected = max(ranked, key=lambda item: item[0])[1] if ranked else None
            path = (selected.skill_id,) if selected else ()
        elif self.config.method == "recursive":
            route = "recursive"
            candidates, path, nodes = self._recursive_rollout(
                task, state, memory, direct, seed + 23
            )

        if should_search:
            selected, candidates, path, nodes = self._search(
                task, state, memory, seed + 31, self.config.search_depth
            )

        selected, safety_fallback = self._shield(task, state, selected)
        if selected is not None:
            memory.last_subtask = selected.skill_id
        calls = self.backend.calls - before_calls
        return Decision(
            selected=selected,
            route=route if not safety_fallback else f"{route}+shield",
            confidence=selected.confidence if selected else 0.0,
            memory_update=memory_update,
            candidates=candidates,
            search_nodes=nodes,
            model_calls=calls,
            latency_ms=(time.perf_counter() - started) * 1000,
            predicted_path=path,
            safety_fallback=safety_fallback,
        )
