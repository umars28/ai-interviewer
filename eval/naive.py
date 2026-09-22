import argparse
import statistics
import sys
from pathlib import Path

from pydantic import BaseModel

from eval.metrics import score_hidden_facts
from interviewer import prompts, views
from interviewer.llm import LLMClient, Role, build_client, user
from interviewer.persona import Persona, load_personas, synthetic_respondent
from interviewer.state import InterviewState, Speaker, Turn, initial_state

DEFAULT_GOAL = "why people abandon the note-taking app they were using"
DEFAULT_TURNS = 6


class NaiveQuestion(BaseModel):
    question: str


def naive_interview(
    research_goal: str,
    persona: Persona,
    client: LLMClient,
    turns: int,
) -> list[Turn]:
    respond = synthetic_respondent(persona, client)
    state: InterviewState = initial_state(research_goal)
    transcript: list[Turn] = []

    for _ in range(turns):
        rendered = prompts.render(
            "naive_interviewer",
            research_goal=research_goal,
            transcript=views.transcript_view(transcript),
        )
        planned = client.structured(Role.INTERVIEWER, [user(rendered)], NaiveQuestion)
        question = planned.question.strip()

        index = len(transcript)
        transcript.append(Turn(index=index, speaker=Speaker.INTERVIEWER, text=question))
        state["transcript"] = transcript
        answer = respond(question, state).strip()
        transcript.append(Turn(index=index + 1, speaker=Speaker.RESPONDENT, text=answer))
        state["transcript"] = transcript

    return transcript


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval.naive")
    parser.add_argument("--personas", type=Path, default=Path("personas"))
    parser.add_argument("--goal", default=DEFAULT_GOAL)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--turns", type=int, default=DEFAULT_TURNS)
    args = parser.parse_args(argv)

    client = build_client()
    if hasattr(client, "check_ready"):
        client.check_ready()

    personas = load_personas(args.personas)
    scores: dict[str, list[float]] = {p.id: [] for p in personas}

    for run_index in range(1, args.repeat + 1):
        for persona in personas:
            print(f"naive interview {persona.id} ({run_index}/{args.repeat}) ...", flush=True)
            transcript = naive_interview(args.goal, persona, client, args.turns)
            facts = score_hidden_facts(persona, transcript)
            recall = sum(1 for f in facts if f.surfaced) / len(facts) if facts else 0.0
            scores[persona.id].append(recall)
            print(f"  recall {recall:.2f} over {args.turns} turns")

    header = f"{'persona':<10} {'runs':>5} {'mean':>7} {'min':>6} {'max':>6}"
    print()
    print(f"Naive interviewer, {args.turns} turns, no graph and no critic")
    print(header)
    print("-" * len(header))
    every = []
    for persona_id, values in scores.items():
        every.extend(values)
        print(
            f"{persona_id:<10} {len(values):>5} {statistics.mean(values):>7.2f} "
            f"{min(values):>6.2f} {max(values):>6.2f}"
        )
    print("-" * len(header))
    print(
        f"overall mean {statistics.mean(every):.2f} across {len(every)} runs, "
        f"range {min(every):.2f} to {max(every):.2f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
