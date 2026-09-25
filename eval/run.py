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


def spread(results: list[InterviewMetrics]) -> str:
    by_persona: dict[str, list[InterviewMetrics]] = {}
    for r in results:
        by_persona.setdefault(r.persona_id, []).append(r)

    header = f"{'persona':<10} {'runs':>5} {'valid':>6} {'recall mean':>12} {'min':>6} {'max':>6}"
    lines = ["", "Across repeats:", header, "-" * len(header)]
    every_valid = []
    for persona_id, runs in by_persona.items():
        valid = [r.hidden_fact_recall for r in runs if not r.degenerate]
        every_valid.extend(valid)
        if valid:
            mean = sum(valid) / len(valid)
            lines.append(
                f"{persona_id:<10} {len(runs):>5} {len(valid):>6} {mean:>12.2f} "
                f"{min(valid):>6.2f} {max(valid):>6.2f}"
            )
        else:
            lines.append(f"{persona_id:<10} {len(runs):>5} {0:>6} {'-':>12} {'-':>6} {'-':>6}")

    lines.append("-" * len(header))
    if len(every_valid) < 2:
        lines.append("too few valid runs to say anything about spread")
        return "\n".join(lines)

    overall = sum(every_valid) / len(every_valid)
    lines.append(
        f"overall mean {overall:.2f} across {len(every_valid)} valid runs, "
        f"range {min(every_valid):.2f} to {max(every_valid):.2f}"
    )
    if max(every_valid) - min(every_valid) >= 0.3:
        lines.append(
            "spread is wide: a single run cannot show whether a change helped. "
            "Compare means over repeats, not one number against another."
        )
    return "\n".join(lines)


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
            f"{r.persona_id:<10} {recall:>7} {r.turn_count:>6} {to_cov!s:>7} "
            f"{r.distinct_question_ratio:>9.2f} {r.forced_fallbacks:>9} "
            f"{len(r.leaked_questions):>7} {stop:<10}"
        )

    if not results:
        return "\n".join(lines)

    trusted = [r.hidden_fact_recall for r in results if not r.degenerate]
    lines.append("-" * len(header))
    if trusted:
        lines.append(
            f"mean recall {sum(trusted) / len(trusted):.2f} over {len(trusted)} valid runs"
        )
    else:
        lines.append("no valid runs: every interview degenerated, so recall means nothing")

    degenerate = [r.persona_id for r in results if r.degenerate]
    if degenerate:
        lines.append(
            f"degenerate: {', '.join(degenerate)} (repeated questions or forced fallbacks)"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval.run")
    parser.add_argument("--personas", type=Path, default=Path("personas"))
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--goal", default=DEFAULT_GOAL)
    parser.add_argument("--only", default=None)
    parser.add_argument("--repeat", type=int, default=1)
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

    try:
        for run_index in range(1, args.repeat + 1):
            for persona in personas:
                label = (
                    f"{persona.id} ({run_index}/{args.repeat})" if args.repeat > 1 else persona.id
                )
                print(f"interviewing {label} ...", flush=True)
                state = run_interview(args.goal, client, synthetic_respondent(persona, client))
                save(state, args.out, f"{persona.id}-r{run_index}")
                metrics = evaluate(persona, state)
                results.append(metrics)
                flag = " [degenerate]" if metrics.degenerate else ""
                print(
                    f"  recall {metrics.hidden_fact_recall:.2f} "
                    f"over {metrics.turn_count} turns{flag}"
                )

        (args.out / "metrics.json").write_text(
            json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8"
        )
        print()
        print(report(results))
        if args.repeat > 1:
            print(spread(results))
    finally:
        if hasattr(client, "usage_summary"):
            print()
            print(client.usage_summary(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
