import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from interviewer.graph import run_interview
from interviewer.llm import build_client
from interviewer.state import InterviewState


def terminal_respondent(question: str, _state: InterviewState) -> str:
    print(f"\n  {question}\n")
    try:
        answer = input("  > ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  (ending interview)")
        return "I need to stop here."
    return answer or "I would rather not say."


def save(state: InterviewState, out_dir: Path, label: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{label}-{stamp}.json"
    payload = {
        "research_goal": state["research_goal"],
        "stop_reason": state.get("stop_reason"),
        "turn_count": state.get("turn_count"),
        "forced_fallbacks": state.get("forced_fallbacks"),
        "goals": [goal.model_dump() for goal in state.get("goals", [])],
        "facts": [fact.model_dump() for fact in state.get("facts", [])],
        "emergent": state.get("emergent", []),
        "rejections": [r.model_dump() for r in state.get("rejection_log", [])],
        "transcript": [turn.model_dump() for turn in state.get("transcript", [])],
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="interviewer.cli")
    parser.add_argument("--goal", required=True)
    parser.add_argument("--out", type=Path, default=Path("transcripts"))
    parser.add_argument("--label", default="interview")
    args = parser.parse_args(argv)

    client = build_client()
    if hasattr(client, "check_ready"):
        client.check_ready()

    print(f"\nResearch goal: {args.goal}")
    print("Answer in your own words. Ctrl-D to stop early.\n")

    try:
        final = run_interview(args.goal, client, terminal_respondent)
        print(f"\n  Stopped: {final.get('stop_reason')} after {final.get('turn_count')} turns")
        print(f"  Saved: {save(final, args.out, args.label)}\n")
    finally:
        if hasattr(client, "usage_summary"):
            print(client.usage_summary(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
