import re
from dataclasses import dataclass, field

from interviewer import rules
from interviewer.persona import Persona
from interviewer.state import GoalStatus, InterviewState, Speaker, Turn

STOPWORDS = frozenset(
    """
    a an and are as at be been because before but by did do does for from had has have
    her his how in into is it its of on or she that the their them then there these they
    this to was were what when where which who why will with would you your about after
    still never about more most some very just also than too own
    """.split()
)

NUMBER_WORDS = frozenset(
    """
    one two three four five six seven eight nine ten twelve fourteen twenty thirty forty
    fifty sixty hundred thousand
    """.split()
)

TOKEN = re.compile(r"[a-z0-9]+")
SURFACED_THRESHOLD = 0.5


def distinctive_tokens(text: str) -> set[str]:
    tokens = TOKEN.findall(text.casefold())
    return {t for t in tokens if t not in STOPWORDS and (len(t) > 3 or t.isdigit() or t in NUMBER_WORDS)}


def respondent_text(transcript: list[Turn]) -> str:
    return " ".join(turn.text for turn in transcript if turn.speaker is Speaker.RESPONDENT)


@dataclass
class FactScore:
    fact: str
    score: float
    surfaced: bool
    missing: list[str] = field(default_factory=list)


def score_hidden_facts(persona: Persona, transcript: list[Turn]) -> list[FactScore]:
    said = distinctive_tokens(respondent_text(transcript))
    scores = []
    for fact in persona.hidden_facts:
        wanted = distinctive_tokens(fact)
        if not wanted:
            continue
        hit = wanted & said
        score = len(hit) / len(wanted)
        scores.append(
            FactScore(
                fact=fact,
                score=round(score, 3),
                surfaced=score >= SURFACED_THRESHOLD,
                missing=sorted(wanted - hit),
            )
        )
    return scores


def questions_asked(transcript: list[Turn]) -> list[str]:
    return [turn.text for turn in transcript if turn.speaker is Speaker.INTERVIEWER]


def rule_violations_reaching_respondent(transcript: list[Turn]) -> list[tuple[str, list[str]]]:
    leaked = []
    for question in questions_asked(transcript):
        violations = rules.check(question)
        if violations:
            leaked.append((question, violations))
    return leaked


def turns_to_coverage(state: InterviewState) -> int | None:
    goals = state.get("goals", [])
    if not goals or any(goal.status is not GoalStatus.COVERED for goal in goals):
        return None
    return state.get("turn_count")


@dataclass
class InterviewMetrics:
    persona_id: str
    stop_reason: str | None
    turn_count: int
    turns_to_coverage: int | None
    hidden_fact_recall: float
    fact_scores: list[FactScore]
    leaked_questions: list[tuple[str, list[str]]]
    forced_fallbacks: int
    goals_covered: int
    goals_total: int

    @property
    def leaked_rate(self) -> float:
        total = self.turn_count or 1
        return round(len(self.leaked_questions) / total, 3)


def evaluate(persona: Persona, state: InterviewState) -> InterviewMetrics:
    transcript = state.get("transcript", [])
    scores = score_hidden_facts(persona, transcript)
    surfaced = sum(1 for s in scores if s.surfaced)
    goals = state.get("goals", [])

    return InterviewMetrics(
        persona_id=persona.id,
        stop_reason=str(state.get("stop_reason")) if state.get("stop_reason") else None,
        turn_count=state.get("turn_count", 0),
        turns_to_coverage=turns_to_coverage(state),
        hidden_fact_recall=round(surfaced / len(scores), 3) if scores else 0.0,
        fact_scores=scores,
        leaked_questions=rule_violations_reaching_respondent(transcript),
        forced_fallbacks=state.get("forced_fallbacks", 0),
        goals_covered=sum(1 for g in goals if g.status is GoalStatus.COVERED),
        goals_total=len(goals),
    )
