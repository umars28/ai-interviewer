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
    header = (
        f"{'persona':<10} {'recall':>7} {'turns':>6} {'to_cov':>7} "
        f"{'distinct':>9} {'fallback':>9} {'leaked':>7} {'stop':<10}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        to_cov = r.turns_to_coverage if r.turns_to_coverage is not None else "-"
        stop = (r.stop_reason or "-").removeprefix("StopReason.")
        recall = "  void" if r.degenerate else f"{r.hidden_fact_recall:>7.2f}"
        lines.append(
            f"{r.persona_id:<10} {recall:>7} {r.turn_count:>6} {str(to_cov):>7} "
            f"{r.distinct_question_ratio:>9.2f} {r.forced_fallbacks:>9} "
            f"{len(r.leaked_questions):>7} {stop:<10}"
        )

    if not results:
        return "\n".join(lines)

    trusted = [r.hidden_fact_recall for r in results if not r.degenerate]
    lines.append("-" * len(header))
    if trusted:
        lines.append(f"mean recall {sum(trusted) / len(trusted):.2f} over {len(trusted)} valid runs")
    else:
        lines.append("no valid runs: every interview degenerated, so recall means nothing")

    degenerate = [r.persona_id for r in results if r.degenerate]
    if degenerate:
        lines.append(f"degenerate: {', '.join(degenerate)} (repeated questions or forced fallbacks)")
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
