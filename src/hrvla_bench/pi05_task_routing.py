"""Explicit language route for the released PI0.5 PickPlaceBox checkpoint.

The upstream HTTP server recognizes ``ppbox`` but not the full Gym task ID's
``pickplacebox`` spelling.  Without this route it passes the Gym ID itself to
the language-conditioned policy.  Keep this exact correction separate from
the locked upstream checkout and from the original rollout evidence.
"""

from __future__ import annotations

from collections.abc import Mapping


PICKPLACE_BOX_GYM_ID = "Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby"
PICKPLACE_BOX_POLICY_TASK = "HOI_pp_box"


def locked_task_route(
    task_value: str | None,
    instructions: Mapping[str, str],
) -> tuple[str, str] | None:
    """Return the exact published task prompt only for the misrouted Gym ID."""

    if task_value != PICKPLACE_BOX_GYM_ID:
        return None
    instruction = instructions.get(PICKPLACE_BOX_POLICY_TASK)
    if not instruction:
        raise ValueError("Released PI0.5 server lacks the HOI_pp_box instruction")
    return PICKPLACE_BOX_POLICY_TASK, instruction
