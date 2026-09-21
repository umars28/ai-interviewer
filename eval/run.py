import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from eval.metrics import InterviewMetrics, evaluate
from interviewer.cli import save
from interviewer.graph import run_interview
from interviewer.llm import build_client
from interviewer.persona import load_personas, synthetic_respondent

DEFAULT_GOAL = "why people abandon the note-taking app they were using"


def report(results: list[InterviewMetrics]) -> str:
    header = f"{'persona':<10} {'recall':>7} {'turns':>6} {'to_cov':>7} {'leaked':>7} {'stop':<12}"
    lines = [header, "-" * len(header)]
    for r in results:
        to_cov = r.turns_to_coverage if r.turns_to_coverage is not None else "-"
        stop = (r.stop_reason or "-").removeprefix("StopReason.")
        lines.append(
            f"{r.persona_id:<10} {r.hidden_fact_recall:>7.2f} {r.turn_count:>6} "
            f"{str(to_cov):>7} {len(r.leaked_questions):>7} {stop:<12}"
        )
    if results:
        mean = sum(r.hidden_fact_recall for r in results) / len(results)
        leaked = sum(len(r.leaked_questions) for r in results)
        lines += ["-" * len(header), f"mean recall {mean:.2f}    leaked questions {leaked}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval.run")
    parser.add_argument("--personas", type=Path, default=Path("personas"))
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--goal", default=DEFAULT_GOAL)
    parser.add_argument("--only", default=None)
    args = parser.parse_args(argv)

    client = build_client()
    if hasattr(client, "check_ready"):
        client.check_ready()

    personas = load_personas(args.personas)
    if args.only:
        personas = [p for p in personas if p.id == args.only]
        if not personas:
            print(f"no persona with id {args.only!r}", file=sys.stderr)
            return 1

    args.out.mkdir(parents=True, exist_ok=True)
    results: list[InterviewMetrics] = []

    for persona in personas:
        print(f"interviewing {persona.id} ...", flush=True)
        state = run_interview(args.goal, client, synthetic_respondent(persona, client))
        save(state, args.out, persona.id)
        metrics = evaluate(persona, state)
        results.append(metrics)
        print(f"  recall {metrics.hidden_fact_recall:.2f} over {metrics.turn_count} turns")

    (args.out / "metrics.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8"
    )
    print()
    print(report(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
