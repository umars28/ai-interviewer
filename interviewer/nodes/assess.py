from interviewer import prompts, views
from interviewer.llm import LLMClient, Role, user
from interviewer.state import (
    AnswerKind,
    Assessment,
    Fact,
    GoalStatus,
    InterviewState,
)


def assess(state: InterviewState, client: LLMClient) -> InterviewState:
    transcript = state.get("transcript", [])
    answer_turn = views.last_respondent_turn(transcript)
    if answer_turn is None:
        return {}

    goal = views.goal_by_id(state, state.get("active_goal_id"))
    rendered = prompts.render(
        "assess",
        goal=goal.question if goal else state["research_goal"],
        question=state.get("pending_question") or "",
        answer=answer_turn.text,
    )
    assessment = client.structured(Role.ASSESSOR, [user(rendered)], Assessment)

    facts = [
        *state.get("facts", []),
        *(Fact(text=text, turn_index=answer_turn.index) for text in assessment.facts),
    ]

    goals = state.get("goals", [])
    if goal is not None:
        goals = [
            g.model_copy(update={"status": _merge(g.status, assessment.goal_progress)})
            if g.id == goal.id
            else g
            for g in goals
        ]

    depth = dict(state.get("probe_depth", {}))
    if goal is not None:
        if state.get("probing"):
            depth[goal.id] = depth.get(goal.id, 0) + 1
        elif assessment.kind is AnswerKind.CONCRETE:
            depth[goal.id] = 0

    emergent = list(state.get("emergent", []))
    if assessment.emergent_topic and assessment.emergent_topic not in emergent:
        emergent.append(assessment.emergent_topic)

    return {
        "last_assessment": assessment,
        "facts": facts,
        "goals": goals,
        "probe_depth": depth,
        "emergent": emergent,
    }


_RANK = {GoalStatus.UNTOUCHED: 0, GoalStatus.SHALLOW: 1, GoalStatus.COVERED: 2}


def _merge(current: GoalStatus, reported: GoalStatus) -> GoalStatus:
    return current if _RANK[current] >= _RANK[reported] else reported
