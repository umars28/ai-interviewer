from interviewer import prompts
from interviewer.llm import LLMClient, Role, user
from interviewer.state import GoalPlan, GoalStatus, InterviewState


def prepare(state: InterviewState, client: LLMClient) -> InterviewState:
    rendered = prompts.render("prepare", research_goal=state["research_goal"])
    plan = client.structured(Role.INTERVIEWER, [user(rendered)], GoalPlan)
    goals = [goal.model_copy(update={"status": GoalStatus.UNTOUCHED}) for goal in plan.goals]
    return {"goals": goals, "probe_depth": {goal.id: 0 for goal in goals}}
