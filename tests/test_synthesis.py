from interviewer.llm.base import Role
from interviewer.llm.fake import FakeClient
from interviewer.state import Speaker, Turn
from interviewer.synthesis import (
    Contradiction,
    Evidence,
    Interview,
    Synthesis,
    Theme,
    synthesize,
)

MIRA_SAID = "Sync had been failing silently for three weeks and I lost the glossary."
DEVAN_SAID = "It took fourteen seconds to open a note while three people waited."

INTERVIEWS = [
    Interview(
        respondent_id="mira",
        transcript=[
            Turn(index=0, speaker=Speaker.INTERVIEWER, text="What happened?"),
            Turn(index=1, speaker=Speaker.RESPONDENT, text=MIRA_SAID),
        ],
    ),
    Interview(
        respondent_id="devan",
        transcript=[
            Turn(index=0, speaker=Speaker.INTERVIEWER, text="What happened?"),
            Turn(index=1, speaker=Speaker.RESPONDENT, text=DEVAN_SAID),
        ],
    ),
]

GOAL = "why people abandon their note-taking app"


def theme(claim: str, respondent_id: str, turn_index: int, quote: str) -> Theme:
    return Theme(
        claim=claim,
        evidence=[Evidence(respondent_id=respondent_id, turn_index=turn_index, quote=quote)],
    )


def test_a_theme_with_a_real_quote_survives():
    client = FakeClient()
    client.script(
        Role.SYNTHESIZER,
        Synthesis(themes=[theme("Silent failure precedes the switch", "mira", 1, "failing silently for three weeks")]),
    )

    report = synthesize(GOAL, INTERVIEWS, client, retries=0)

    assert len(report.themes) == 1
    assert report.quote_validity == 1.0


def test_a_theme_whose_quote_was_invented_is_dropped():
    client = FakeClient()
    client.script(
        Role.SYNTHESIZER,
        Synthesis(themes=[theme("Users blame themselves", "mira", 1, "I assumed it was my fault")]),
    )

    report = synthesize(GOAL, INTERVIEWS, client, retries=0)

    assert report.themes == []
    assert report.quote_validity == 0.0
    assert "not found" in report.rejected[0][1]


def test_a_quote_attributed_to_the_wrong_respondent_is_dropped():
    client = FakeClient()
    client.script(
        Role.SYNTHESIZER,
        Synthesis(themes=[theme("Latency drives churn", "mira", 1, "fourteen seconds to open a note")]),
    )

    report = synthesize(GOAL, INTERVIEWS, client, retries=0)

    assert report.themes == []


def test_an_unknown_respondent_is_rejected():
    client = FakeClient()
    client.script(
        Role.SYNTHESIZER,
        Synthesis(themes=[theme("Something", "hanna", 1, "anything at all")]),
    )

    report = synthesize(GOAL, INTERVIEWS, client, retries=0)

    assert "unknown respondent" in report.rejected[0][1]


def test_a_theme_keeps_its_valid_evidence_and_loses_the_rest():
    client = FakeClient()
    client.script(
        Role.SYNTHESIZER,
        Synthesis(
            themes=[
                Theme(
                    claim="Loss of trust precedes the switch",
                    evidence=[
                        Evidence(respondent_id="mira", turn_index=1, quote="I lost the glossary"),
                        Evidence(respondent_id="devan", turn_index=1, quote="I stopped trusting it"),
                    ],
                )
            ]
        ),
    )

    report = synthesize(GOAL, INTERVIEWS, client, retries=0)

    assert len(report.themes) == 1
    assert len(report.themes[0].evidence) == 1
    assert report.quote_validity == 0.5


def test_rejected_quotes_trigger_one_retry_with_the_failures_named():
    client = FakeClient()
    client.script(
        Role.SYNTHESIZER,
        Synthesis(themes=[theme("Guessed", "mira", 1, "a quote nobody said")]),
        Synthesis(themes=[theme("Corrected", "mira", 1, "I lost the glossary")]),
    )

    report = synthesize(GOAL, INTERVIEWS, client, retries=1)

    assert len(report.themes) == 1
    assert report.themes[0].claim == "Corrected"
    second_call = client.calls_for(Role.SYNTHESIZER)[1]
    assert "a quote nobody said" in second_call[-1]["content"]


def test_a_contradiction_needs_every_quote_to_check_out():
    client = FakeClient()
    client.script(
        Role.SYNTHESIZER,
        Synthesis(
            contradictions=[
                Contradiction(
                    description="One blames sync, the other blames latency",
                    evidence=[
                        Evidence(respondent_id="mira", turn_index=1, quote="failing silently"),
                        Evidence(respondent_id="devan", turn_index=1, quote="it was never slow"),
                    ],
                )
            ]
        ),
    )

    report = synthesize(GOAL, INTERVIEWS, client, retries=0)

    assert report.contradictions == []


def test_a_fully_evidenced_contradiction_is_kept():
    client = FakeClient()
    client.script(
        Role.SYNTHESIZER,
        Synthesis(
            contradictions=[
                Contradiction(
                    description="Different failure modes",
                    evidence=[
                        Evidence(respondent_id="mira", turn_index=1, quote="failing silently"),
                        Evidence(respondent_id="devan", turn_index=1, quote="fourteen seconds"),
                    ],
                )
            ]
        ),
    )

    report = synthesize(GOAL, INTERVIEWS, client, retries=0)

    assert len(report.contradictions) == 1
