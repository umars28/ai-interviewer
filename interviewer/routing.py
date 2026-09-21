from dataclasses import dataclass

from interviewer.state import (
    FATIGUE_MIN_CHARS,
    FATIGUE_WINDOW,
    MAX_PROBE_DEPTH,
    MAX_TURNS,
    AnswerKind,
    Goal,
    GoalStatus,
    InterviewState,
    StopReason,
)

CONTINUE = "plan_question"
WRAPUP = "wrapup"


def all_covered(goals: list[Goal]) -> bool:
    return bool(goals) and all(goal.status is GoalStatus.COVERED for goal in goals)


def is_fatigued(recent_lengths: list[int]) -> bool:
    window = recent_lengths[-FATIGUE_WINDOW:]
    if len(window) < FATIGUE_WINDOW:
        return False
    if any(length >= FATIGUE_MIN_CHARS for length in window):
        return False
    return window == sorted(window, reverse=True)


def should_stop(state: InterviewState) -> StopReason | None:
    if all_covered(state.get("goals", [])):
        return StopReason.COVERAGE
    if state.get("turn_count", 0) >= MAX_TURNS:
        return StopReason.TURN_CAP
    if is_fatigued(state.get("recent_lengths", [])):
        return StopReason.FATIGUE
    return None


def route_after_assess(state: InterviewState) -> str:
    return WRAPUP if should_stop(state) is not None else CONTINUE


@dataclass(frozen=True)
class Focus:
    goal_id: str | None
    probing: bool
    reason: str


def _worth_probing(state: InterviewState) -> bool:
    assessment = state.get("last_assessment")
    if assessment is None:
        return False
    return assessment.kind in (AnswerKind.VAGUE, AnswerKind.EVASIVE)


def select_focus(state: InterviewState) -> Focus:
    goals = state.get("goals", [])
    active_id = state.get("active_goal_id")
    depth = state.get("probe_depth", {})

    if active_id and _worth_probing(state):
        if depth.get(active_id, 0) < MAX_PROBE_DEPTH:
            return Focus(active_id, True, "answer was unclear and probe budget remains")
        return Focus(_next_open_goal(goals, active_id), False, "probe budget exhausted")

    if active_id:
        active = next((goal for goal in goals if goal.id == active_id), None)
        if active is not None and active.status is not GoalStatus.COVERED:
            return Focus(active_id, False, "goal still open")

    return Focus(_next_open_goal(goals, active_id), False, "moving to next open goal")


def _next_open_goal(goals: list[Goal], after_id: str | None) -> str | None:
    open_goals = [goal for goal in goals if goal.status is not GoalStatus.COVERED]
    if not open_goals:
        return None
    if after_id is None:
        return open_goals[0].id
    remaining = [goal for goal in open_goals if goal.id != after_id]
    return remaining[0].id if remaining else open_goals[0].id
