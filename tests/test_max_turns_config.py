import importlib
import os

from interviewer import state


def reload_state(monkeypatch, value: str | None):
    if value is None:
        monkeypatch.delenv("INTERVIEW_MAX_TURNS", raising=False)
    else:
        monkeypatch.setenv("INTERVIEW_MAX_TURNS", value)
    return importlib.reload(state)


def test_the_default_turn_cap_is_unchanged_when_no_override_is_set(monkeypatch):
    reloaded = reload_state(monkeypatch, None)

    assert reloaded.MAX_TURNS == reloaded.DEFAULT_MAX_TURNS == 25


def test_an_env_override_lowers_the_turn_cap(monkeypatch):
    reloaded = reload_state(monkeypatch, "8")

    assert reloaded.MAX_TURNS == 8


def teardown_module():
    os.environ.pop("INTERVIEW_MAX_TURNS", None)
    importlib.reload(state)
