from interviewer import prompts, rules, views
from interviewer.llm import LLMClient, Role, user
from interviewer.state import (
    MAX_CRITIC_REJECTIONS,
    GoalStatus,
    InterviewState,
    Rejection,
    Verdict,
)

PASS = "ask"
REWRITE = "plan_question"


def critic(state: InterviewState, client: LLMClient) -> InterviewState:
    question = state.get("pending_question")
    if not question:
        return {"critic_feedback": "no question was produced", "critic_rejections": 1}

    asked = views.asked_questions(state.get("transcript", []))
    mechanical = rules.check(question, asked)
    if mechanical:
        return _reject(
            state,
            Verdict(passed=False, violations=mechanical, feedback=rules.describe(mechanical)),
            source="rules",
        )

    covered = [goal for goal in state.get("goals", []) if goal.status is GoalStatus.COVERED]
    rendered = prompts.render(
        "critic",
        covered_goals=views.goals_view(covered),
        asked_questions=views.list_view(views.asked_questions(state.get("transcript", []))),
        question=question,
    )

    verdict = client.structured(Role.CRITIC, [user(rendered)], Verdict)
    if verdict.passed:
        return {"critic_feedback": None, "critic_rejections": 0, "last_verdict": verdict}
    return _reject(state, verdict, source="model")


def _reject(state: InterviewState, verdict: Verdict, source: str) -> InterviewState:
    violations = ", ".join(verdict.violations) or "unspecified"
    feedback = f"{violations}. {verdict.feedback}".strip()
    entry = Rejection(
        question=state.get("pending_question") or "",
        source=source,
        violations=verdict.violations,
        feedback=verdict.feedback,
    )
    return {
        "critic_feedback": feedback,
        "critic_rejections": state.get("critic_rejections", 0) + 1,
        "last_verdict": verdict,
        "rejection_log": [*state.get("rejection_log", []), entry],
    }


def route_after_critic(state: InterviewState) -> str:
    if state.get("critic_feedback") is None:
        return PASS
    if state.get("critic_rejections", 0) >= MAX_CRITIC_REJECTIONS:
        return PASS
    return REWRITE
