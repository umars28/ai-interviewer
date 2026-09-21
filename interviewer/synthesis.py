from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from interviewer import prompts
from interviewer.llm import LLMClient, Role, user
from interviewer.quotes import check_against_transcript
from interviewer.state import Turn
from interviewer.views import transcript_view


class Evidence(BaseModel):
    respondent_id: str
    turn_index: int
    quote: str


class Theme(BaseModel):
    claim: str
    evidence: list[Evidence] = Field(default_factory=list)


class Contradiction(BaseModel):
    description: str
    evidence: list[Evidence] = Field(default_factory=list)


class Synthesis(BaseModel):
    themes: list[Theme] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)


@dataclass(frozen=True)
class Interview:
    respondent_id: str
    transcript: list[Turn]


@dataclass
class SynthesisReport:
    themes: list[Theme] = field(default_factory=list)
    contradictions: list[Contradiction] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)

    @property
    def quote_validity(self) -> float:
        kept = sum(len(theme.evidence) for theme in self.themes)
        total = kept + len(self.rejected)
        return 1.0 if total == 0 else round(kept / total, 3)


def verify(evidence: Evidence, interviews: list[Interview]) -> str | None:
    interview = next((i for i in interviews if i.respondent_id == evidence.respondent_id), None)
    if interview is None:
        return f"unknown respondent {evidence.respondent_id!r}"
    result = check_against_transcript(evidence.quote, evidence.turn_index, interview.transcript)
    return None if result.valid else result.reason


def _render(research_goal: str, interviews: list[Interview]) -> str:
    blocks = [
        f"### respondent: {i.respondent_id}\n{transcript_view(i.transcript)}" for i in interviews
    ]
    return prompts.render(
        "synthesize",
        research_goal=research_goal,
        transcripts="\n\n".join(blocks),
    )


def synthesize(
    research_goal: str,
    interviews: list[Interview],
    client: LLMClient,
    retries: int = 1,
) -> SynthesisReport:
    rendered = _render(research_goal, interviews)
    report = SynthesisReport()

    for attempt in range(retries + 1):
        messages = [user(rendered)]
        if attempt and report.rejected:
            failures = "\n".join(f"- {quote!r}: {reason}" for quote, reason in report.rejected)
            messages.append(
                user(
                    "These quotes did not match the transcripts character for character:\n"
                    f"{failures}\n\nRedo the write-up. Copy quotes exactly or drop the theme."
                )
            )

        result = client.structured(Role.SYNTHESIZER, messages, Synthesis)
        report = _filter(result, interviews)
        if not report.rejected:
            break

    return report


def _filter(result: Synthesis, interviews: list[Interview]) -> SynthesisReport:
    report = SynthesisReport()

    for theme in result.themes:
        kept = []
        for evidence in theme.evidence:
            reason = verify(evidence, interviews)
            if reason is None:
                kept.append(evidence)
            else:
                report.rejected.append((evidence.quote, reason))
        if kept:
            report.themes.append(Theme(claim=theme.claim, evidence=kept))

    for contradiction in result.contradictions:
        kept = [e for e in contradiction.evidence if verify(e, interviews) is None]
        if len(kept) == len(contradiction.evidence) and kept:
            report.contradictions.append(
                Contradiction(description=contradiction.description, evidence=kept)
            )

    return report
