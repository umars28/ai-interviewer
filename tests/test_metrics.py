from eval.metrics import (
    distinct_question_ratio,
    distinctive_tokens,
    evaluate,
    rule_violations_reaching_respondent,
    score_hidden_facts,
    turns_to_coverage,
)
from interviewer import rules
from interviewer.persona import Persona
from interviewer.state import Goal, GoalStatus, Speaker, StopReason, Turn, initial_state

PERSONA = Persona(
    id="mira",
    background="a translator",
    voice="dry",
    knowledge="switched apps last year",
    hidden_facts=[
        "She spent about forty minutes searching before she accepted the file was gone.",
        "She still pays for the old subscription because she cannot find where to cancel it.",
    ],
)


def interview(*texts: str) -> list[Turn]:
    turns = []
    for i, text in enumerate(texts):
        speaker = Speaker.INTERVIEWER if i % 2 == 0 else Speaker.RESPONDENT
        turns.append(Turn(index=i, speaker=speaker, text=text))
    return turns


def test_distinctive_tokens_drop_filler_and_keep_numbers():
    tokens = distinctive_tokens("She spent about forty minutes searching for it")
    assert "forty" in tokens
    assert "minutes" in tokens
    assert "she" not in tokens
    assert "for" not in tokens


def test_a_planted_fact_counts_when_the_respondent_says_it():
    transcript = interview(
        "How long did you search?",
        "I spent forty minutes searching before I accepted the file was gone.",
    )
    scores = score_hidden_facts(PERSONA, transcript)
    assert scores[0].surfaced
    assert not scores[1].surfaced


def test_the_interviewer_saying_it_does_not_count():
    transcript = interview(
        "Did you spend forty minutes searching before the file was gone?",
        "Something like that.",
    )
    assert not score_hidden_facts(PERSONA, transcript)[0].surfaced


def test_missing_tokens_are_reported_for_inspection():
    transcript = interview("How long?", "About forty minutes, I think.")
    score = score_hidden_facts(PERSONA, transcript)[0]
    assert "searching" in score.missing


def test_leaked_rule_breaking_questions_are_detected():
    transcript = interview(
        "Would you use a faster app?",
        "Maybe.",
        "What happened the last time it failed?",
        "It lost my glossary.",
    )
    leaked = rule_violations_reaching_respondent(transcript)
    assert len(leaked) == 1
    assert "hypothetical" in leaked[0][1]


def test_turns_to_coverage_is_none_while_a_goal_is_open():
    state = initial_state("goal")
    state["goals"] = [
        Goal(id="g1", question="q", status=GoalStatus.COVERED),
        Goal(id="g2", question="q", status=GoalStatus.SHALLOW),
    ]
    state["turn_count"] = 7
    assert turns_to_coverage(state) is None


def test_turns_to_coverage_reports_the_turn_count_once_covered():
    state = initial_state("goal")
    state["goals"] = [Goal(id="g1", question="q", status=GoalStatus.COVERED)]
    state["turn_count"] = 7
    assert turns_to_coverage(state) == 7


def test_evaluate_assembles_the_four_numbers():
    state = initial_state("goal")
    state["goals"] = [Goal(id="g1", question="q", status=GoalStatus.COVERED)]
    state["turn_count"] = 4
    state["stop_reason"] = StopReason.COVERAGE
    state["transcript"] = interview(
        "What happened?",
        "I spent forty minutes searching before I accepted the file was gone.",
        "Would you switch again?",
        "Probably not.",
    )

    metrics = evaluate(PERSONA, state)

    assert metrics.persona_id == "mira"
    assert metrics.hidden_fact_recall == 0.5
    assert metrics.turns_to_coverage == 4
    assert metrics.goals_covered == metrics.goals_total == 1
    assert len(metrics.leaked_questions) == 1
    assert metrics.leaked_rate == 0.25


def test_the_wrapup_summary_is_not_counted_as_a_leaked_question():
    from interviewer.nodes.wrapup import TEMPLATE

    summary = TEMPLATE.format(facts="- (turn 1) She searched for forty minutes.")
    assert rules.check(summary), "the summary is deliberately long, so the rules should flag it"

    transcript = [
        Turn(index=0, speaker=Speaker.INTERVIEWER, text="What happened the last time?"),
        Turn(index=1, speaker=Speaker.RESPONDENT, text="The sync failed."),
        Turn(index=2, speaker=Speaker.INTERVIEWER, text=summary, vetted=False),
        Turn(index=3, speaker=Speaker.RESPONDENT, text="That is right."),
    ]

    assert rule_violations_reaching_respondent(transcript) == []
    assert distinct_question_ratio(transcript) == 1.0


def test_an_unvetted_marker_does_not_hide_a_real_leak():
    transcript = [
        Turn(index=0, speaker=Speaker.INTERVIEWER, text="Would you use a faster app?"),
        Turn(index=1, speaker=Speaker.RESPONDENT, text="Maybe."),
    ]
    assert len(rule_violations_reaching_respondent(transcript)) == 1
