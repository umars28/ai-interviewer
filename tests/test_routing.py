from interviewer.routing import (
    CONTINUE,
    WRAPUP,
    is_fatigued,
    route_after_assess,
    select_focus,
    should_stop,
)
from interviewer.state import (
    MAX_PROBE_DEPTH,
    MAX_TURNS,
    AnswerKind,
    Assessment,
    Goal,
    GoalStatus,
    InterviewState,
    StopReason,
    initial_state,
)


def state_with(**overrides) -> InterviewState:
    state = initial_state("why people abandon their note-taking app")
    state.update(overrides)
    return state


def goals(*statuses: GoalStatus) -> list[Goal]:
    return [
        Goal(id=f"g{i + 1}", question=f"question {i + 1}", status=status)
        for i, status in enumerate(statuses)
    ]


def assessment(kind: AnswerKind) -> Assessment:
    return Assessment(kind=kind, goal_progress=GoalStatus.SHALLOW)


def test_stops_when_every_goal_is_covered():
    state = state_with(goals=goals(GoalStatus.COVERED, GoalStatus.COVERED))
    assert should_stop(state) is StopReason.COVERAGE
    assert route_after_assess(state) == WRAPUP


def test_does_not_stop_when_one_goal_is_open():
    state = state_with(goals=goals(GoalStatus.COVERED, GoalStatus.SHALLOW))
    assert should_stop(state) is None
    assert route_after_assess(state) == CONTINUE


def test_empty_goal_list_is_not_treated_as_covered():
    assert should_stop(state_with(goals=[])) is None


def test_turn_cap_stops_the_interview():
    state = state_with(goals=goals(GoalStatus.SHALLOW), turn_count=MAX_TURNS)
    assert should_stop(state) is StopReason.TURN_CAP


def test_coverage_wins_over_turn_cap():
    state = state_with(goals=goals(GoalStatus.COVERED), turn_count=MAX_TURNS)
    assert should_stop(state) is StopReason.COVERAGE


def test_fatigue_needs_a_full_window():
    assert not is_fatigued([10, 8])


def test_fatigue_detects_short_and_shrinking_answers():
    assert is_fatigued([30, 20, 5])


def test_one_long_answer_in_the_window_clears_fatigue():
    assert not is_fatigued([30, 200, 5])


def test_short_but_not_shrinking_answers_are_not_fatigue():
    assert not is_fatigued([5, 30, 10])


def test_vague_answer_triggers_a_probe_on_the_same_goal():
    state = state_with(
        goals=goals(GoalStatus.SHALLOW, GoalStatus.UNTOUCHED),
        active_goal_id="g1",
        last_assessment=assessment(AnswerKind.VAGUE),
        probe_depth={"g1": 1},
    )
    focus = select_focus(state)
    assert focus.goal_id == "g1"
    assert focus.probing


def test_evasive_answer_also_triggers_a_probe():
    state = state_with(
        goals=goals(GoalStatus.SHALLOW),
        active_goal_id="g1",
        last_assessment=assessment(AnswerKind.EVASIVE),
        probe_depth={"g1": 0},
    )
    assert select_focus(state).probing


def test_probe_budget_stops_the_interrogation():
    state = state_with(
        goals=goals(GoalStatus.SHALLOW, GoalStatus.UNTOUCHED),
        active_goal_id="g1",
        last_assessment=assessment(AnswerKind.VAGUE),
        probe_depth={"g1": MAX_PROBE_DEPTH},
    )
    focus = select_focus(state)
    assert not focus.probing
    assert focus.goal_id == "g2"


def test_concrete_answer_keeps_an_open_goal_without_probing():
    state = state_with(
        goals=goals(GoalStatus.SHALLOW, GoalStatus.UNTOUCHED),
        active_goal_id="g1",
        last_assessment=assessment(AnswerKind.CONCRETE),
    )
    focus = select_focus(state)
    assert focus.goal_id == "g1"
    assert not focus.probing


def test_covered_goal_hands_off_to_the_next_open_goal():
    state = state_with(
        goals=goals(GoalStatus.COVERED, GoalStatus.UNTOUCHED),
        active_goal_id="g1",
        last_assessment=assessment(AnswerKind.CONCRETE),
    )
    assert select_focus(state).goal_id == "g2"


def test_first_focus_is_the_first_open_goal():
    state = state_with(goals=goals(GoalStatus.UNTOUCHED, GoalStatus.UNTOUCHED))
    assert select_focus(state).goal_id == "g1"


def test_no_open_goals_yields_no_focus():
    state = state_with(goals=goals(GoalStatus.COVERED), active_goal_id="g1")
    assert select_focus(state).goal_id is None
