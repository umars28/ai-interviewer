from enum import StrEnum
from typing import TypedDict

from pydantic import BaseModel, Field


class GoalStatus(StrEnum):
    UNTOUCHED = "untouched"
    SHALLOW = "shallow"
    COVERED = "covered"


class Speaker(StrEnum):
    INTERVIEWER = "interviewer"
    RESPONDENT = "respondent"


class AnswerKind(StrEnum):
    CONCRETE = "concrete"
    VAGUE = "vague"
    EVASIVE = "evasive"
    NEW_THREAD = "new_thread"


class StopReason(StrEnum):
    COVERAGE = "coverage"
    FATIGUE = "fatigue"
    TURN_CAP = "turn_cap"
    STALLED = "stalled"


class Goal(BaseModel):
    id: str
    question: str
    status: GoalStatus = GoalStatus.UNTOUCHED


class Turn(BaseModel):
    index: int
    speaker: Speaker
    text: str


class Fact(BaseModel):
    text: str
    turn_index: int


class Assessment(BaseModel):
    kind: AnswerKind
    facts: list[str] = Field(default_factory=list)
    goal_progress: GoalStatus
    emergent_topic: str | None = None


class Verdict(BaseModel):
    passed: bool
    violations: list[str] = Field(default_factory=list)
    feedback: str = ""


class Rejection(BaseModel):
    question: str
    source: str
    violations: list[str]
    feedback: str


class PlannedQuestion(BaseModel):
    goal_id: str
    question: str


class GoalPlan(BaseModel):
    goals: list[Goal]


class InterviewState(TypedDict, total=False):
    research_goal: str
    goals: list[Goal]
    transcript: list[Turn]
    facts: list[Fact]
    probe_depth: dict[str, int]
    emergent: list[str]
    recent_lengths: list[int]
    turn_count: int
    active_goal_id: str | None
    probing: bool
    pending_question: str | None
    critic_feedback: str | None
    critic_rejections: int
    last_verdict: Verdict | None
    rejection_log: list[Rejection]
    forced_fallbacks: int
    last_assessment: Assessment | None
    stop_reason: StopReason | None


MAX_TURNS = 25
MAX_PROBE_DEPTH = 3
MAX_CRITIC_REJECTIONS = 3
MAX_FORCED_FALLBACKS = 3
FATIGUE_WINDOW = 3
FATIGUE_MIN_CHARS = 40


def initial_state(research_goal: str) -> InterviewState:
    return InterviewState(
        research_goal=research_goal,
        goals=[],
        transcript=[],
        facts=[],
        probe_depth={},
        emergent=[],
        recent_lengths=[],
        turn_count=0,
        active_goal_id=None,
        probing=False,
        pending_question=None,
        critic_feedback=None,
        critic_rejections=0,
        last_verdict=None,
        rejection_log=[],
        forced_fallbacks=0,
        last_assessment=None,
        stop_reason=None,
    )
