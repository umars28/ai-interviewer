import pytest

from interviewer.quotes import check_against_transcript, check_quote, locate_quote, normalize
from interviewer.state import Speaker, Turn

SOURCE = (
    "I was looking for the meeting notes from the Tuesday standup and I just "
    "could not find them — I searched for maybe two minutes and then gave up."
)


def transcript() -> list[Turn]:
    return [
        Turn(
            index=0, speaker=Speaker.INTERVIEWER, text="Tell me about the last time you switched."
        ),
        Turn(index=1, speaker=Speaker.RESPONDENT, text=SOURCE),
        Turn(index=2, speaker=Speaker.INTERVIEWER, text="How long did you search?"),
    ]


def test_exact_substring_passes():
    assert check_quote("I searched for maybe two minutes", SOURCE)


def test_whole_source_passes():
    assert check_quote(SOURCE, SOURCE)


def test_fabricated_quote_fails():
    result = check_quote("I gave up after ten seconds", SOURCE)
    assert not result
    assert "not found" in result.reason


def test_plausible_paraphrase_fails():
    assert not check_quote("I couldn't find my meeting notes", SOURCE)


def test_curly_quotes_and_dashes_normalize():
    assert check_quote("could not find them — I searched", SOURCE)
    assert check_quote("could not find them - I searched", SOURCE)


def test_collapsed_whitespace_normalizes():
    assert check_quote("I   searched\nfor  maybe\ttwo minutes", SOURCE)


def test_case_is_ignored():
    assert check_quote("I SEARCHED FOR MAYBE TWO MINUTES", SOURCE)


def test_elided_quote_passes_when_fragments_are_in_order():
    assert check_quote("I was looking for the meeting notes ... and then gave up", SOURCE)


def test_elided_quote_fails_when_fragments_are_out_of_order():
    result = check_quote("and then gave up ... I was looking for the meeting notes", SOURCE)
    assert not result
    assert "out of order" in result.reason


def test_ellipsis_cannot_stitch_unrelated_text():
    assert not check_quote("I was looking for ... a refund", SOURCE)


@pytest.mark.parametrize("quote", ["", "   ", "...", "…"])
def test_empty_and_ellipsis_only_quotes_fail(quote):
    assert not check_quote(quote, SOURCE)


def test_transcript_check_uses_the_named_turn():
    assert check_against_transcript("I searched for maybe two minutes", 1, transcript())


def test_transcript_check_fails_when_quote_is_from_another_turn():
    result = check_against_transcript("How long did you search?", 1, transcript())
    assert not result


def test_transcript_check_fails_on_unknown_turn_index():
    result = check_against_transcript("anything", 99, transcript())
    assert not result
    assert "no turn with index" in result.reason


def test_locate_quote_finds_the_owning_turn():
    assert locate_quote("I searched for maybe two minutes", transcript()) == 1


def test_locate_quote_returns_none_for_fabrication():
    assert locate_quote("I never used that app", transcript()) is None


def test_normalize_is_idempotent():
    once = normalize(SOURCE)
    assert normalize(once) == once
