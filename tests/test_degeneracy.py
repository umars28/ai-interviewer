from eval.metrics import distinct_question_ratio, evaluate
from interviewer import rules
from interviewer.nodes.plan_question import FALLBACK_QUESTION
from interviewer.persona import Persona
from interviewer.routing import should_stop
from interviewer.state import (
    MAX_FORCED_FALLBACKS,
    Goal,
    GoalStatus,
    Speaker,
    StopReason,
    Turn,
    initial_state,
)

PERSONA = Persona(
    id="mira",
    background="a translator",
    voice="dry",
    knowledge="switched apps",
    hidden_facts=["She spent about forty minutes searching before she gave up."],
)


def test_polite_invitations_are_not_treated_as_closed_questions():
    for question in [
        "Can you tell me what happened the last time you lost a file?",
        "Could you describe the last time sync failed?",
        "Can you walk me through what you did next?",
    ]:
        assert rules.check(question) == []


def test_genuinely_closed_questions_are_still_caught():
    assert "closed" in rules.check("Can you use the app offline?")
    assert "closed" in rules.check("Was the old app slow?")


def test_asking_the_same_question_twice_is_a_violation():
    asked = ["What happened the last time you lost a note?"]
    assert rules.check("what happened the last time you lost a note?", asked) == ["repeat"]


def test_a_differently_worded_question_is_not_a_repeat():
    asked = ["What happened the last time you lost a note?"]
    assert rules.check("How long did you search for it?", asked) == []


def test_repeated_fallbacks_stall_the_interview():
    state = initial_state("goal")
    state["goals"] = [Goal(id="g1", question="q", status=GoalStatus.SHALLOW)]
    state["forced_fallbacks"] = MAX_FORCED_FALLBACKS
    assert should_stop(state) is StopReason.STALLED


def test_distinct_question_ratio_exposes_a_loop():
    transcript = []
    for i in range(6):
        transcript.append(Turn(index=2 * i, speaker=Speaker.INTERVIEWER, text=FALLBACK_QUESTION))
        transcript.append(Turn(index=2 * i + 1, speaker=Speaker.RESPONDENT, text="Same as before."))
    assert distinct_question_ratio(transcript) < 0.2


def test_a_looping_interview_cannot_score_recall_even_when_facts_leak():
    state = initial_state("goal")
    state["goals"] = [Goal(id="g1", question="q", status=GoalStatus.COVERED)]
    state["turn_count"] = 6
    state["forced_fallbacks"] = 6
    state["stop_reason"] = StopReason.COVERAGE
    transcript = []
    for i in range(6):
        transcript.append(Turn(index=2 * i, speaker=Speaker.INTERVIEWER, text=FALLBACK_QUESTION))
        transcript.append(
            Turn(
                index=2 * i + 1,
                speaker=Speaker.RESPONDENT,
                text="I spent about forty minutes searching before I gave up.",
            )
        )
    state["transcript"] = transcript

    metrics = evaluate(PERSONA, state)

    assert metrics.hidden_fact_recall == 1.0
    assert metrics.degenerate
    assert metrics.trustworthy_recall is None


def test_a_varied_interview_keeps_its_recall():
    state = initial_state("goal")
    state["goals"] = [Goal(id="g1", question="q", status=GoalStatus.COVERED)]
    state["turn_count"] = 2
    state["stop_reason"] = StopReason.COVERAGE
    state["transcript"] = [
        Turn(index=0, speaker=Speaker.INTERVIEWER, text="What happened the last time?"),
        Turn(index=1, speaker=Speaker.RESPONDENT, text="The sync failed."),
        Turn(index=2, speaker=Speaker.INTERVIEWER, text="How long did you search for it?"),
        Turn(
            index=3,
            speaker=Speaker.RESPONDENT,
            text="I spent about forty minutes searching before I gave up.",
        ),
    ]

    metrics = evaluate(PERSONA, state)

    assert not metrics.degenerate
    assert metrics.trustworthy_recall == 1.0
