from interviewer.state import Fact, Goal, InterviewState, Speaker, Turn

NOTHING = "(nothing yet)"


def goals_view(goals: list[Goal]) -> str:
    if not goals:
        return NOTHING
    return "\n".join(f"- [{goal.status}] {goal.id}: {goal.question}" for goal in goals)


def transcript_view(transcript: list[Turn], limit: int | None = None) -> str:
    turns = transcript[-limit:] if limit else transcript
    if not turns:
        return NOTHING
    label = {Speaker.INTERVIEWER: "Interviewer", Speaker.RESPONDENT: "Respondent"}
    return "\n".join(f"[{turn.index}] {label[turn.speaker]}: {turn.text}" for turn in turns)


def facts_view(facts: list[Fact]) -> str:
    if not facts:
        return NOTHING
    return "\n".join(f"- (turn {fact.turn_index}) {fact.text}" for fact in facts)


def list_view(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else NOTHING


def asked_questions(transcript: list[Turn]) -> list[str]:
    return [turn.text for turn in transcript if turn.speaker is Speaker.INTERVIEWER]


def goal_by_id(state: InterviewState, goal_id: str | None) -> Goal | None:
    if goal_id is None:
        return None
    return next((goal for goal in state.get("goals", []) if goal.id == goal_id), None)


def last_respondent_turn(transcript: list[Turn]) -> Turn | None:
    for turn in reversed(transcript):
        if turn.speaker is Speaker.RESPONDENT:
            return turn
    return None
