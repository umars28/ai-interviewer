from eval.naive import NaiveQuestion, naive_interview
from interviewer.llm.base import Role
from interviewer.llm.fake import FakeClient
from interviewer.persona import Persona, Reply
from interviewer.state import Speaker

PERSONA = Persona(
    id="mira",
    background="a translator",
    voice="dry",
    knowledge="switched apps",
    hidden_facts=["She searched for forty minutes."],
)


def counting_client() -> FakeClient:
    questions = iter(range(1, 100))
    client = FakeClient()
    client.respond_with(
        Role.INTERVIEWER,
        lambda _m, _s: NaiveQuestion(question=f"Question {next(questions)}?"),
    )
    client.respond_with(Role.RESPONDENT, lambda _m, _s: Reply(text="I do not recall."))
    return client


def test_the_control_runs_exactly_the_turn_budget_it_is_given():
    transcript = naive_interview("goal", PERSONA, counting_client(), turns=4)

    assert len(transcript) == 8
    assert [t.speaker for t in transcript[:2]] == [Speaker.INTERVIEWER, Speaker.RESPONDENT]


def test_the_control_indexes_turns_contiguously_like_the_graph():
    transcript = naive_interview("goal", PERSONA, counting_client(), turns=3)

    assert [t.index for t in transcript] == list(range(6))


def test_the_control_uses_the_same_interviewer_and_respondent_roles():
    client = counting_client()
    naive_interview("goal", PERSONA, client, turns=2)

    assert len(client.calls_for(Role.INTERVIEWER)) == 2
    assert len(client.calls_for(Role.RESPONDENT)) == 2
    assert client.calls_for(Role.CRITIC) == []
    assert client.calls_for(Role.ASSESSOR) == []


def test_the_control_shows_the_interviewer_the_conversation_so_far():
    client = counting_client()
    naive_interview("goal", PERSONA, client, turns=3)

    last_prompt = client.calls_for(Role.INTERVIEWER)[-1][0]["content"]
    assert "Question 1?" in last_prompt
    assert "I do not recall." in last_prompt
