from functools import cache
from pathlib import Path

PROMPT_DIR = Path(__file__).parent


@cache
def load(name: str) -> str:
    path = PROMPT_DIR / f"{name}.md"
    if not path.exists():
        available = ", ".join(sorted(p.stem for p in PROMPT_DIR.glob("*.md")))
        raise FileNotFoundError(f"no prompt named {name!r}; available: {available}")
    return path.read_text(encoding="utf-8")


def render(name: str, **fields: object) -> str:
    return load(name).format(**fields)
