from collections.abc import Callable

from interviewer.nodes.plan_question import FALLBACK_QUESTION
from interviewer.state import MAX_CRITIC_REJECTIONS, InterviewState, Speaker, Turn

Respondent = Callable[[str, InterviewState], str]


def ask(state: InterviewState, respondent: Respondent) -> InterviewState:
    question = state.get("pending_question") or FALLBACK_QUESTION
    forced = 0

    if (
        state.get("critic_feedback") is not None
        and state.get("critic_rejections", 0) >= MAX_CRITIC_REJECTIONS
    ):
        question = FALLBACK_QUESTION
        forced = 1

    transcript = list(state.get("transcript", []))
    next_index = len(transcript)
    transcript.append(Turn(index=next_index, speaker=Speaker.INTERVIEWER, text=question))

    answer = respondent(question, state).strip()
    transcript.append(Turn(index=next_index + 1, speaker=Speaker.RESPONDENT, text=answer))

    return {
        "transcript": transcript,
        "pending_question": question,
        "recent_lengths": [*state.get("recent_lengths", []), len(answer)],
        "turn_count": state.get("turn_count", 0) + 1,
        "critic_feedback": None,
        "critic_rejections": 0,
        "forced_fallbacks": state.get("forced_fallbacks", 0) + forced,
    }
