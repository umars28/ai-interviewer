import pytest

from interviewer import rules

GOOD = [
    "What happened the last time you looked for a note and could not find it?",
    "Tell me about the most recent time you switched apps.",
    "Walk me through what you did after the sync failed.",
    "How long did you spend searching before you gave up?",
    "Why did you stop using it?",
]

BAD = {
    "Would you use a faster note app?": "hypothetical",
    "If you had a better search, how would that change things?": "hypothetical",
    "Was the old app slow?": "closed",
    "Did that frustrate you?": "closed",
    "What do most people do when sync fails?": "speculative_about_others",
    "When did you switch and why did you pick this one?": "double",
    "What happened? And how did you feel about it?": "double",
}


@pytest.mark.parametrize("question", GOOD)
def test_well_formed_questions_pass(question):
    assert rules.check(question) == []


@pytest.mark.parametrize(("question", "violation"), BAD.items())
def test_malformed_questions_are_caught(question, violation):
    assert violation in rules.check(question)


def test_a_question_can_break_several_rules_at_once():
    violations = rules.check("Would you use a faster app, and did the old one frustrate you?")
    assert "hypothetical" in violations
    assert "double" in violations


def test_long_preamble_is_rejected():
    question = (
        "I want to understand your workflow in detail because it helps us design better. "
        "There are many possible angles here and I do not want to miss any of them. "
        "So what happened?"
    )
    assert "too_long" in rules.check(question)


def test_and_inside_a_single_question_is_not_double_barrelled():
    assert rules.check("What happened between the sync failure and the deadline?") == []


def test_describe_turns_violations_into_feedback():
    text = rules.describe(["hypothetical", "double"])
    assert "would do" in text
    assert "two questions" in text


RECOUNTS = [
    "How would you describe your overall experience with the app?",
    "How would you characterise the search before you left?",
    "What would you say was the turning point?",
    "How would you sum up the last month of using it?",
]


@pytest.mark.parametrize("question", RECOUNTS)
def test_asking_someone_to_describe_their_own_experience_is_not_hypothetical(question):
    assert "hypothetical" not in rules.check(question)


STILL_HYPOTHETICAL = [
    "How would you use a faster app?",
    "What would you do if sync failed again?",
    "How would you feel about paying for it?",
]


@pytest.mark.parametrize("question", STILL_HYPOTHETICAL)
def test_genuinely_hypothetical_questions_are_still_caught(question):
    assert "hypothetical" in rules.check(question)
