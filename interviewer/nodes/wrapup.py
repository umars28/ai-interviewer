from interviewer.nodes.ask import Respondent
from interviewer.routing import should_stop
from interviewer.state import InterviewState, Speaker, Turn
from interviewer.views import facts_view

TEMPLATE = (
    "Before we finish, let me check I got this right.\n\n{facts}\n\n"
    "Is any of that wrong, or did I miss something that mattered?"
)


def wrapup(state: InterviewState, respondent: Respondent) -> InterviewState:
    transcript = list(state.get("transcript", []))
    summary = TEMPLATE.format(facts=facts_view(state.get("facts", [])))

    next_index = len(transcript)
    transcript.append(
        Turn(index=next_index, speaker=Speaker.INTERVIEWER, text=summary, vetted=False)
    )
    correction = respondent(summary, state).strip()
    transcript.append(Turn(index=next_index + 1, speaker=Speaker.RESPONDENT, text=correction))

    return {
        "transcript": transcript,
        "stop_reason": should_stop(state),
        "pending_question": None,
    }
