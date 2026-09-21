from interviewer import prompts, views
from interviewer.llm import LLMClient, Role, user
from interviewer.routing import select_focus
from interviewer.state import InterviewState, PlannedQuestion

FALLBACK_QUESTION = "Tell me about the last time that came up. What happened?"


def plan_question(state: InterviewState, client: LLMClient) -> InterviewState:
    focus = select_focus(state)
    goal = views.goal_by_id(state, focus.goal_id)
    depth = state.get("probe_depth", {}).get(focus.goal_id or "", 0)

    probe_note = ""
    if focus.probing:
        probe_note = (
            f"The last answer was unclear. Probe the vague part of it. "
            f"This is follow-up {depth + 1} on this goal; after three you must move on."
        )

    feedback = state.get("critic_feedback")
    critic_note = f"A reviewer rejected your previous attempt: {feedback}" if feedback else ""

    rendered = prompts.render(
        "plan_question",
        research_goal=state["research_goal"],
        goals=views.goals_view(state.get("goals", [])),
        transcript=views.transcript_view(state.get("transcript", []), limit=8),
        facts=views.facts_view(state.get("facts", [])),
        emergent=views.list_view(state.get("emergent", [])),
        focus=goal.question if goal else "wrap up; every goal is covered",
        probe_note=probe_note,
        critic_feedback=critic_note,
    )

    planned = client.structured(Role.INTERVIEWER, [user(rendered)], PlannedQuestion)
    question = planned.question.strip() or FALLBACK_QUESTION
    goal_id = planned.goal_id if views.goal_by_id(state, planned.goal_id) else focus.goal_id

    return {
        "pending_question": question,
        "active_goal_id": goal_id,
        "probing": focus.probing,
    }
