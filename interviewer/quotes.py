import re
import unicodedata
from dataclasses import dataclass

from interviewer.state import Turn

_CURLY = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "–": "-",
        "—": "-",
        " ": " ",
    }
)

_ELLIPSIS = re.compile(r"\s*(?:\.{3,}|…)\s*")
_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text).translate(_CURLY)
    return _WHITESPACE.sub(" ", folded).strip().casefold()


@dataclass(frozen=True)
class QuoteCheck:
    valid: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.valid


def check_quote(quote: str, source: str) -> QuoteCheck:
    if not quote.strip():
        return QuoteCheck(False, "quote is empty")

    haystack = normalize(source)
    fragments = [normalize(part) for part in _ELLIPSIS.split(quote) if part.strip()]
    if not fragments:
        return QuoteCheck(False, "quote is only an ellipsis")

    cursor = 0
    for fragment in fragments:
        found = haystack.find(fragment, cursor)
        if found == -1:
            if fragment in haystack:
                return QuoteCheck(False, f"fragment out of order in source: {fragment!r}")
            return QuoteCheck(False, f"fragment not found in source: {fragment!r}")
        cursor = found + len(fragment)
    return QuoteCheck(True)


def check_against_transcript(quote: str, turn_index: int, transcript: list[Turn]) -> QuoteCheck:
    turn = next((t for t in transcript if t.index == turn_index), None)
    if turn is None:
        return QuoteCheck(False, f"no turn with index {turn_index}")
    return check_quote(quote, turn.text)


def locate_quote(quote: str, transcript: list[Turn]) -> int | None:
    for turn in transcript:
        if check_quote(quote, turn.text):
            return turn.index
    return None
