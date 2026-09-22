import re

TOKEN = re.compile(r"[a-z0-9]+")

HYPOTHETICAL = re.compile(
    r"\b(would you|will you|could you see yourself|do you think you.d|if you had|"
    r"suppose you|imagine you|might you)\b",
    re.IGNORECASE,
)

CLOSED_OPENERS = re.compile(
    r"^\s*(was|were|did|do|does|is|are|have|has|had|can|could|should|would|will)\b",
    re.IGNORECASE,
)

INVITATION_VERBS = r"(tell|describe|walk|explain|share|talk|say|give|show|take|sum|summarise|summarize|characterise|characterize)"

INVITATION = re.compile(
    rf"^\s*(can|could|would|will)\s+you\s+(please\s+)?{INVITATION_VERBS}\b",
    re.IGNORECASE,
)

RECOUNT = re.compile(
    rf"\b(how|what)\s+would\s+you\s+(please\s+)?{INVITATION_VERBS}\b",
    re.IGNORECASE,
)

OTHERS = re.compile(
    r"\b(most people|other users|people generally|users usually|everyone else)\b",
    re.IGNORECASE,
)

SENTENCE_END = re.compile(r"[.!?]+(?:\s|$)")
QUESTION_WORD = re.compile(
    r"\b(what|why|how|when|where|who|which|tell me|describe|walk me)\b", re.IGNORECASE
)
WH_OPENER = re.compile(r"^(what|why|how|when|where|who|which)\b", re.IGNORECASE)
AUX_WITH_SUBJECT = re.compile(
    r"^(was|were|did|do|does|is|are|have|has|had|can|could|should|would|will)\s+"
    r"(i|you|he|she|it|we|they|that|this|the|your|my|his|her|their|there)\b",
    re.IGNORECASE,
)


def _opens_a_question(clause: str) -> bool:
    text = clause.strip()
    return bool(WH_OPENER.match(text) or AUX_WITH_SUBJECT.match(text))


MAX_SENTENCES = 2
MAX_WORDS = 45


def sentence_count(text: str) -> int:
    return len([part for part in SENTENCE_END.split(text) if part.strip()]) or 1


def is_double_barrelled(text: str) -> bool:
    body = text.strip().rstrip("?")
    if "?" in body:
        return True
    for joiner in (" and ", " or "):
        head, sep, tail = body.partition(joiner)
        if not sep:
            continue
        head_is_question = bool(QUESTION_WORD.search(head) or _opens_a_question(head))
        if head_is_question and _opens_a_question(tail):
            return True
    return False


def fingerprint(question: str) -> str:
    return " ".join(TOKEN.findall(question.casefold()))


def check(question: str, asked: list[str] | tuple[str, ...] = ()) -> list[str]:
    text = question.strip()
    violations: list[str] = []

    if HYPOTHETICAL.search(text) and not RECOUNT.search(text):
        violations.append("hypothetical")
    if CLOSED_OPENERS.match(text) and not INVITATION.match(text):
        violations.append("closed")
    if OTHERS.search(text):
        violations.append("speculative_about_others")
    if is_double_barrelled(text):
        violations.append("double")
    if sentence_count(text) > MAX_SENTENCES or len(text.split()) > MAX_WORDS:
        violations.append("too_long")
    if fingerprint(text) in {fingerprint(prior) for prior in asked}:
        violations.append("repeat")

    return violations


def describe(violations: list[str]) -> str:
    explanations = {
        "hypothetical": "asks what they would do rather than what they did",
        "closed": "opens with a yes/no verb instead of what, why, how, or tell me about",
        "double": "packs two questions into one, so one of them gets lost",
        "too_long": "too long; the question is buried",
        "speculative_about_others": "asks them to speak for other people",
        "repeat": "already asked in this interview, word for word",
    }
    return "; ".join(explanations.get(v, v) for v in violations)
