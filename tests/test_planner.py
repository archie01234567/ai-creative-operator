"""Structured outputs guarantee shape, not rules. The planner must catch rule violations."""

import pytest

from creative_operator.errors import PlanningError
from creative_operator.models import Storyboard, Strategy
from creative_operator.planner import plan_storyboard, plan_strategy
from creative_operator.prompt_builder import build_tasks
from creative_operator.providers.base import Capabilities


class ScriptedLLM:
    supports_vision = False

    def __init__(self, *answers):
        self.answers = list(answers)
        self.prompts = []

    def structured(self, *, system, user, schema, images=(), effort="high"):
        self.prompts.append(user)
        return schema.model_validate(self.answers.pop(0))


def test_storyboard_outside_duration_window_is_repaired_once_then_fails(brief, fixture_json):
    strategy = Strategy.model_validate(fixture_json("strategy"))
    good = fixture_json("storyboard")
    too_long = {**good, "scenes": good["scenes"] * 2}  # 26s, 8 scenes

    llm = ScriptedLLM(too_long, good)
    board = plan_storyboard(llm, brief, strategy)
    assert board.total_seconds == 13
    assert "sum to 26s" in llm.prompts[1], "the repair prompt must say what was wrong"

    with pytest.raises(PlanningError, match="10-15s"):
        plan_storyboard(ScriptedLLM(too_long, too_long), brief, strategy)


def test_strategy_must_select_an_angle_that_exists(brief, fixture_json):
    good = fixture_json("strategy")
    with pytest.raises(PlanningError, match="not one of"):
        plan_strategy(ScriptedLLM(*[{**good, "selected_angle": "Something Else"}] * 2), brief)
    with pytest.raises(PlanningError, match="at least 3"):
        plan_strategy(ScriptedLLM(*[{**good, "angles": good["angles"][:2]}] * 2), brief)


def test_render_plan_adapts_to_provider_capabilities(fixture_json):
    board = Storyboard.model_validate(fixture_json("storyboard"))

    mode, tasks = build_tasks(board, Capabilities(3, 15, True))
    assert mode == "single" and len(tasks) == 1 and tasks[0].duration == 13
    assert all(f"SHOT {i}" in tasks[0].prompt for i in range(1, 5))

    # An 8-second provider cannot do the spot in one take: one request per shot, later shots
    # carry the identity reference, and nothing is requested below the provider minimum.
    mode, tasks = build_tasks(board, Capabilities(4, 8, True), feedback=["hands look warped"])
    assert mode == "per_scene" and len(tasks) == 4
    assert [t.uses_reference for t in tasks] == [False, True, True, True]
    assert all(t.duration >= 4 for t in tasks)
    assert all(board.protagonist in t.prompt and "hands look warped" in t.prompt for t in tasks)
