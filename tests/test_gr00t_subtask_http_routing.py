"""ST changes GR00T language only when grounded state is supplied."""

import base64

import numpy as np
import pytest

from hrvla_bench.humanoidarena_bridge import ACTION_FIELDS
from hrvla_bench.humanoidarena_gr00t_http import HumanoidArenaGr00tState
from hrvla_subtask.http_routing import ObservedStateRouter


PROGRAM = {
    "schema_version": 1,
    "tasks": [{
        "task_id": "test_task",
        "goal_instruction": "Finish the task.",
        "initial_state": ["start"],
        "goal_state": ["done"],
        "max_steps": 4,
        "skills": [{
            "id": "finish", "instruction": "Complete the observed task step.",
            "requires": ["start"], "adds": ["done"],
        }],
    }],
}


class FakePolicy:
    def reset(self, options=None):
        return {}

    def get_action(self, observation):
        self.observation = observation
        return {
            field.name: np.zeros((1, 40, field.width), dtype=np.float32)
            for field in ACTION_FIELDS
        }


def payload():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    return {
        "task": "test_task",
        "observation": {
            "images": {"front": {
                "shape": list(image.shape), "dtype": "uint8",
                "data_b64": base64.b64encode(image.tobytes()).decode(),
            }},
            "state": np.zeros(64).tolist(),
        },
    }


def test_routes_grounded_subtask_into_gr00t_language_only():
    policy = FakePolicy()
    state = HumanoidArenaGr00tState(policy, instruction_router=ObservedStateRouter(PROGRAM))
    state.reset(7)
    request = payload()
    request["observed_predicates"] = ["start"]
    result = state.infer_payload(request)
    assert result["selected_skill_id"] == "finish"
    assert policy.observation["language"]["annotation.human.task_description"] == [
        ["Complete the observed task step."]
    ]
    assert result["action_chunk"] == [[0.0] * 40] * 40


def test_rejects_missing_or_unknown_detector_state():
    state = HumanoidArenaGr00tState(FakePolicy(), instruction_router=ObservedStateRouter(PROGRAM))
    request = payload()
    with pytest.raises(ValueError, match="observed_predicates"):
        state.infer_payload(request)
    request["observed_predicates"] = ["fabricated"]
    with pytest.raises(ValueError, match="unknown observed predicates"):
        state.infer_payload(request)
