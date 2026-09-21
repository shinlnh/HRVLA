from hrvla_bench.pi05_task_routing import (
    PICKPLACE_BOX_GYM_ID,
    locked_task_route,
)


INSTRUCTIONS = {
    "HOI_pp_box": "Move the box from the table onto the shelf.",
    "HSI_boxing": "Strike the green markers on the punching bag.",
}


def test_pickplace_gym_id_uses_released_policy_instruction():
    assert locked_task_route(PICKPLACE_BOX_GYM_ID, INSTRUCTIONS) == (
        "HOI_pp_box",
        "Move the box from the table onto the shelf.",
    )


def test_unrelated_tasks_keep_upstream_routing():
    assert locked_task_route("Isaac-Move-Boxing-Bag-G129-Dex3-Wholebody", INSTRUCTIONS) is None
    assert locked_task_route(None, INSTRUCTIONS) is None


def test_missing_published_instruction_fails_closed():
    try:
        locked_task_route(PICKPLACE_BOX_GYM_ID, {})
    except ValueError as exc:
        assert "HOI_pp_box" in str(exc)
    else:
        raise AssertionError("missing prompt must fail closed")
