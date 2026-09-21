from interviewer.llm.base import Role
from interviewer.llm.fake import FakeClient
from interviewer.nodes.assess import assess
from interviewer.state import (
    AnswerKind,
    Assessment,
    Goal,
    GoalStatus,
    Speaker,
    Turn,
    initial_state,
)


def state_after_one_answer(answer: str = "It took about forty minutes to find it."):
    state = initial_state("why people abandon their note app")
    state["goals"] = [Goal(id="g1", question="what triggered the switch")]
    state["active_goal_id"] = "g1"
    state["pending_question"] = "What happened the last time?"
    state["transcript"] = [
        Turn(index=0, speaker=Speaker.INTERVIEWER, text="What happened the last time?"),
        Turn(index=1, speaker=Speaker.RESPONDENT, text=answer),
    ]
    return state


def test_a_goal_cannot_be_covered_without_a_single_extracted_fact():
    client = FakeClient().script(
        Role.ASSESSOR,
        Assessment(facts=[], kind=AnswerKind.CONCRETE, goal_progress=GoalStatus.COVERED),
    )

    out = assess(state_after_one_answer(), client)

    assert out["goals"][0].status is GoalStatus.SHALLOW
    assert out["facts"] == []


def test_a_goal_with_evidence_is_allowed_to_be_covered():
    client = FakeClient().script(
        Role.ASSESSOR,
        Assessment(
            facts=["She searched for about forty minutes."],
            kind=AnswerKind.CONCRETE,
            goal_progress=GoalStatus.COVERED,
        ),
    )

    out = assess(state_after_one_answer(), client)

    assert out["goals"][0].status is GoalStatus.COVERED
    assert [fact.text for fact in out["facts"]] == ["She searched for about forty minutes."]
    assert out["facts"][0].turn_index == 1


def test_concrete_without_facts_is_reclassified_as_vague():
    client = FakeClient().script(
        Role.ASSESSOR,
        Assessment(facts=[], kind=AnswerKind.CONCRETE, goal_progress=GoalStatus.SHALLOW),
    )

    out = assess(state_after_one_answer(), client)

    assert out["last_assessment"].kind is AnswerKind.VAGUE


def test_reclassified_answers_still_earn_a_probe():
    state = state_after_one_answer()
    state["probing"] = False
    client = FakeClient().script(
        Role.ASSESSOR,
        Assessment(facts=[], kind=AnswerKind.CONCRETE, goal_progress=GoalStatus.SHALLOW),
    )

    out = assess(state, client)
    state.update(out)

    from interviewer.routing import select_focus

    assert select_focus(state).probing


def test_a_genuinely_vague_answer_is_left_alone():
    client = FakeClient().script(
        Role.ASSESSOR,
        Assessment(facts=[], kind=AnswerKind.VAGUE, goal_progress=GoalStatus.SHALLOW),
    )

    out = assess(state_after_one_answer("A while, I suppose."), client)

    assert out["last_assessment"].kind is AnswerKind.VAGUE
    assert out["goals"][0].status is GoalStatus.SHALLOW
