from pathlib import Path

import pytest

from interviewer.persona import Persona, load_personas

REPO_PERSONAS = Path(__file__).resolve().parents[1] / "personas"

VALID = """
id: sample
background: |
  Someone who keeps notes.
voice: |
  Terse.
knowledge: |
  Switched apps last year.
hidden_facts:
  - The trigger was losing a file.
  - She searched for forty minutes.
"""


def write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def test_a_persona_loads_with_its_fields_stripped(tmp_path):
    persona = Persona.from_file(write(tmp_path, "sample.yaml", VALID))

    assert persona.id == "sample"
    assert persona.background == "Someone who keeps notes."
    assert persona.hidden_facts == [
        "The trigger was losing a file.",
        "She searched for forty minutes.",
    ]


def test_the_filename_supplies_the_id_when_the_file_omits_it(tmp_path):
    body = VALID.replace("id: sample\n", "")
    assert Persona.from_file(write(tmp_path, "devan.yaml", body)).id == "devan"


def test_a_missing_field_fails_loudly(tmp_path):
    body = VALID.replace("voice: |\n  Terse.\n", "")
    with pytest.raises(KeyError):
        Persona.from_file(write(tmp_path, "broken.yaml", body))


def test_an_empty_directory_is_an_error_not_an_empty_run(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_personas(tmp_path)


def test_personas_load_in_a_stable_order(tmp_path):
    write(tmp_path, "b.yaml", VALID)
    write(tmp_path, "a.yaml", VALID)
    assert [p.id for p in load_personas(tmp_path)] == ["sample", "sample"]


def test_the_shipped_personas_all_carry_five_planted_facts():
    personas = load_personas(REPO_PERSONAS)

    assert {p.id for p in personas} == {"mira", "devan", "hanna"}
    for persona in personas:
        assert len(persona.hidden_facts) == 5
        assert all(fact.strip() for fact in persona.hidden_facts)
