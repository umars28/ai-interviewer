import pytest

from interviewer import rules
from interviewer.graph import run_interview
from interviewer.llm.base import Role
from interviewer.llm.fake import FakeClient
from interviewer.nodes.plan_question import FALLBACK_QUESTION
from interviewer.state import (
    MAX_PROBE_DEPTH,
    MAX_TURNS,
    AnswerKind,
    Assessment,
    Goal,
    GoalPlan,
    GoalStatus,
    InterviewState,
    PlannedQuestion,
    Speaker,
    StopReason,
    Verdict,
)

GOAL = "why people abandon their note-taking app"


def plan(*ids: str) -> GoalPlan:
    return GoalPlan(goals=[Goal(id=i, question=f"what about {i}") for i in ids])


def question(goal_id: str, text: str = "What happened the last time?") -> PlannedQuestion:
    return PlannedQuestion(goal_id=goal_id, question=text)


def passing() -> Verdict:
    return Verdict(passed=True)


def rejecting(*violations: str) -> Verdict:
    return Verdict(passed=False, violations=list(violations), feedback="anchor to a past episode")


def answered(kind: AnswerKind, progress: GoalStatus, facts: list[str] | None = None) -> Assessment:
    return Assessment(kind=kind, goal_progress=progress, facts=facts or [])


def echo_respondent(text: str) -> object:
    def respond(_question: str, _state: InterviewState) -> str:
        return text

    return respond


def test_interview_finishes_when_every_goal_is_covered():
    client = FakeClient()
    client.script(Role.INTERVIEWER, plan("g1", "g2"), question("g1"), question("g2"))
    client.script(Role.CRITIC, passing(), passing())
    client.script(
        Role.ASSESSOR,
        answered(AnswerKind.CONCRETE, GoalStatus.COVERED, ["searched for two minutes"]),
        answered(AnswerKind.CONCRETE, GoalStatus.COVERED, ["switched in March"]),
    )

    final = run_interview(GOAL, client, echo_respondent("I searched for two minutes."))

    assert final["stop_reason"] is StopReason.COVERAGE
    assert final["turn_count"] == 2
    assert all(goal.status is GoalStatus.COVERED for goal in final["goals"])
    assert [fact.text for fact in final["facts"]] == [
        "searched for two minutes",
        "switched in March",
    ]


def test_transcript_alternates_and_indexes_contiguously():
    client = FakeClient()
    client.script(Role.INTERVIEWER, plan("g1"), question("g1"))
    client.script(Role.CRITIC, passing())
    client.script(Role.ASSESSOR, answered(AnswerKind.CONCRETE, GoalStatus.COVERED))

    final = run_interview(GOAL, client, echo_respondent("It was last Tuesday."))
    transcript = final["transcript"]

    assert [turn.index for turn in transcript] == list(range(len(transcript)))
    assert [turn.speaker for turn in transcript] == [
        Speaker.INTERVIEWER,
        Speaker.RESPONDENT,
        Speaker.INTERVIEWER,
        Speaker.RESPONDENT,
    ]


def test_rule_breaking_question_is_rewritten_without_consulting_the_model():
    client = FakeClient()
    client.script(
        Role.INTERVIEWER,
        plan("g1"),
        question("g1", "Would you use a faster app?"),
        question("g1", "What happened the last time you looked for a note?"),
    )
    client.script(Role.CRITIC, passing())
    client.script(Role.ASSESSOR, answered(AnswerKind.CONCRETE, GoalStatus.COVERED))

    final = run_interview(GOAL, client, echo_respondent("I gave up after two minutes."))
    asked = [turn.text for turn in final["transcript"] if turn.speaker is Speaker.INTERVIEWER]

    assert "Would you use a faster app?" not in asked
    assert "What happened the last time you looked for a note?" in asked
    assert len(client.calls_for(Role.CRITIC)) == 1


def test_subtly_leading_question_is_caught_by_the_model_critic():
    leading = "What made the old app so frustrating for you?"
    assert rules.check(leading) == []

    client = FakeClient()
    client.script(
        Role.INTERVIEWER,
        plan("g1"),
        question("g1", leading),
        question("g1", "What happened the last time you opened the old app?"),
    )
    client.script(Role.CRITIC, rejecting("leading"), passing())
    client.script(Role.ASSESSOR, answered(AnswerKind.CONCRETE, GoalStatus.COVERED))

    final = run_interview(GOAL, client, echo_respondent("It took fourteen seconds to open."))
    asked = [turn.text for turn in final["transcript"] if turn.speaker is Speaker.INTERVIEWER]

    assert leading not in asked
    assert len(client.calls_for(Role.CRITIC)) == 2


def test_a_question_the_critic_never_cleared_is_never_sent():
    client = FakeClient()
    client.script(
        Role.INTERVIEWER,
        plan("g1"),
        *[question("g1", "Was the old app slow?") for _ in range(3)],
    )
    client.script(Role.CRITIC, *[rejecting("leading") for _ in range(3)])
    client.script(Role.ASSESSOR, answered(AnswerKind.CONCRETE, GoalStatus.COVERED))

    final = run_interview(GOAL, client, echo_respondent("Sometimes."))
    asked = [turn.text for turn in final["transcript"] if turn.speaker is Speaker.INTERVIEWER]

    assert "Was the old app slow?" not in asked
    assert FALLBACK_QUESTION in asked
    assert final["forced_fallbacks"] == 1


def test_vague_answers_are_probed_then_the_interview_moves_on():
    client = FakeClient()
    client.script(Role.INTERVIEWER, plan("g1", "g2"))
    client.respond_with(
        Role.INTERVIEWER,
        lambda _messages, _schema: question("g1", "How long exactly?"),
    )
    client.respond_with(Role.CRITIC, lambda _messages, _schema: passing())
    client.respond_with(
        Role.ASSESSOR,
        lambda _messages, _schema: answered(AnswerKind.VAGUE, GoalStatus.SHALLOW),
    )

    vague_but_engaged = "A while, I guess. Honestly I could not tell you how long it took."
    final = run_interview(GOAL, client, echo_respondent(vague_but_engaged))

    assert final["probe_depth"]["g1"] == MAX_PROBE_DEPTH
    assert final["turn_count"] == MAX_TURNS
    assert final["stop_reason"] is StopReason.TURN_CAP


def test_short_shrinking_answers_end_the_interview_early():
    client = FakeClient()
    client.script(Role.INTERVIEWER, plan("g1", "g2"))
    client.respond_with(Role.INTERVIEWER, lambda _m, _s: question("g1"))
    client.respond_with(Role.CRITIC, lambda _m, _s: passing())
    client.respond_with(
        Role.ASSESSOR,
        lambda _m, _s: answered(AnswerKind.CONCRETE, GoalStatus.SHALLOW),
    )

    replies = iter(["A fairly long answer about what happened last Tuesday.", "yeah", "mm", "ok"])

    def fading(_question: str, _state: InterviewState) -> str:
        return next(replies, "ok")

    final = run_interview(GOAL, client, fading)

    assert final["stop_reason"] is StopReason.FATIGUE
    assert final["turn_count"] < MAX_TURNS


def test_wrapup_reads_the_facts_back_to_the_respondent():
    client = FakeClient()
    client.script(Role.INTERVIEWER, plan("g1"), question("g1"))
    client.script(Role.CRITIC, passing())
    client.script(
        Role.ASSESSOR,
        answered(AnswerKind.CONCRETE, GoalStatus.COVERED, ["gave up after two minutes"]),
    )

    final = run_interview(GOAL, client, echo_respondent("Right, two minutes."))
    summary = final["transcript"][-2].text

    assert "gave up after two minutes" in summary
    assert final["transcript"][-1].speaker is Speaker.RESPONDENT


def test_emergent_topics_are_recorded_once():
    client = FakeClient()
    client.script(Role.INTERVIEWER, plan("g1", "g2"), question("g1"), question("g2"))
    client.script(Role.CRITIC, passing(), passing())
    client.script(
        Role.ASSESSOR,
        Assessment(
            kind=AnswerKind.NEW_THREAD,
            goal_progress=GoalStatus.COVERED,
            emergent_topic="shared folders with a team",
        ),
        Assessment(
            kind=AnswerKind.CONCRETE,
            goal_progress=GoalStatus.COVERED,
            emergent_topic="shared folders with a team",
        ),
    )

    final = run_interview(GOAL, client, echo_respondent("We shared a folder."))

    assert final["emergent"] == ["shared folders with a team"]


def test_missing_script_fails_loudly_rather_than_silently():
    client = FakeClient()
    client.script(Role.INTERVIEWER, plan("g1"))

    with pytest.raises(Exception, match="no scripted response"):
        run_interview(GOAL, client, echo_respondent("anything"))
