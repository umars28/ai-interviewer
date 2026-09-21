import argparse
import json
import sys
from pathlib import Path

from interviewer.llm import build_client
from interviewer.state import Speaker, Turn
from interviewer.synthesis import Interview, SynthesisReport, synthesize

DEFAULT_GOAL = "why people abandon the note-taking app they were using"


def load_interviews(directory: Path) -> list[Interview]:
    interviews = []
    for path in sorted(directory.glob("*.json")):
        if path.name == "metrics.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        transcript = [
            Turn(index=t["index"], speaker=Speaker(t["speaker"]), text=t["text"])
            for t in payload["transcript"]
        ]
        interviews.append(Interview(respondent_id=path.stem.split("-")[0], transcript=transcript))
    if not interviews:
        raise FileNotFoundError(f"no transcripts in {directory}")
    return interviews


def render(report: SynthesisReport) -> str:
    lines = [f"Quote validity: {report.quote_validity:.0%}", ""]
    for theme in report.themes:
        lines.append(f"## {theme.claim}")
        for evidence in theme.evidence:
            lines.append(
                f'  - {evidence.respondent_id} (turn {evidence.turn_index}): "{evidence.quote}"'
            )
        lines.append("")
    if report.contradictions:
        lines.append("## Contradictions")
        for contradiction in report.contradictions:
            lines.append(f"- {contradiction.description}")
            for evidence in contradiction.evidence:
                lines.append(
                    f'    {evidence.respondent_id} (turn {evidence.turn_index}): "{evidence.quote}"'
                )
    if report.rejected:
        lines += ["", f"Dropped {len(report.rejected)} unverifiable quotes:"]
        lines += [f"  - {quote!r}: {reason}" for quote, reason in report.rejected]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval.synthesize")
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--goal", default=DEFAULT_GOAL)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    client = build_client()
    interviews = load_interviews(args.runs)
    report = synthesize(args.goal, interviews, client)

    text = render(report)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
