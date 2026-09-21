# ai-interviewer

An agent that **interviews people** instead of answering them.

Give it a research goal. It talks to a respondent over chat, decides when to probe deeper and when to move on, refuses to ask leading questions, and stops when the research goals are covered — not when the model feels done. After several interviews it synthesises themes across respondents, and every theme must carry a quote that provably exists in a transcript.

## Why this exists

Most LLM projects cannot be evaluated: the only judge is "looks good to me". This one can. Synthetic respondents are seeded with facts that only surface under correct probing, which turns interview quality into four numbers:

| Metric | How it is computed |
| --- | --- |
| Hidden fact recall | Of 5 planted facts, how many the agent surfaced. Only counted from the respondent's own turns — the interviewer saying a fact does not score it. |
| Leading question rate | Questions that violated the interview rules but still reached the respondent |
| Turns to coverage | Turns needed before every research goal is covered |
| Quote validity | Must be 100% — checked by string matching, not by a model |
| Distinct question ratio | Guard against the metric lying. See below. |

### Why the recall number can lie

The first live run scored 1.00 recall and looked excellent. The transcript showed the agent
asking one identical fallback question six times while the respondent volunteered every
planted fact unprompted on turn one. The interview was worthless and the metric said it was
perfect.

Recall alone measures what the respondent leaked, not what the interviewer earned. So a run
is marked **degenerate** — and its recall reported as `void` rather than as a number — when
the distinct question ratio drops below 0.8, when forced fallbacks account for half the
turns, or when the graph stops with `stalled`. A metric that cannot be embarrassed by a
broken run is not measuring anything.

## Design

Three parts carry the weight.

**Coverage as state, not vibes.** The graph tracks each research goal as `untouched` / `shallow` / `covered`. That is the stop condition, and it is why this needs a stateful graph rather than one prompt.

**A critic that rejects questions before they are sent.** Hypothetical questions, leading questions, double-barrelled questions, and closed questions where a story is needed are all caught and rewritten. Interview methodology is prescriptive enough to be checked.

**Quote validation in code.** Synthesis output is rejected unless the quote matches the transcript exactly after normalisation. A model cannot talk its way past a string comparison.

Two guardrails keep it usable on real humans: `probe_depth` caps consecutive follow-ups on one topic so it does not read as an interrogation, and shrinking answer lengths trigger a wrap-up.

## Architecture

```
prepare ──▶ plan_question ──▶ critic ──┬─(reject, <3x)─▶ plan_question
                                       │
                                  (pass)
                                       ▼
                                      ask ──▶ assess ──▶ route ──┬─▶ plan_question
                                                                 └─▶ wrapup ──▶ END
```

Nodes are plain Python functions. Models are requested by **role**, never by name, so switching provider touches one registry:

| Role | Ollama (local dev) | Claude (later) |
| --- | --- | --- |
| `INTERVIEWER` | `qwen3:14b` | `claude-opus-5` |
| `CRITIC` | `qwen3:14b` | `claude-opus-5` |
| `ASSESSOR` | `qwen3:8b` | `claude-opus-5` |
| `RESPONDENT` | `qwen3:8b` | `claude-haiku-4-5` |
| `SYNTHESIZER` | `qwen3:14b` | `claude-opus-5` |

## Setup

```sh
uv sync --extra dev
cp .env.example .env
```

Local models via Ollama:

```sh
ollama serve &
ollama pull qwen3:14b
ollama pull qwen3:8b
```

`qwen3:14b` (~9 GB) and `qwen3:8b` (~5 GB) stay resident together on 24 GB.

Two Ollama defaults will quietly degrade the run if left alone:

| Setting | Why it matters |
| --- | --- |
| `OLLAMA_MAX_LOADED_MODELS=2` | The graph alternates between two models every turn. With a limit of one, Ollama unloads and reloads on each switch. No error — the interview just gets many times slower. |
| `OLLAMA_NUM_CTX=16384` | Ollama's default context is 4096 tokens. The interviewer prompt carries the growing transcript and crosses that around turn 15, after which the earliest turns are silently truncated and the agent starts repeating questions it already asked. |

Override models per role with `OLLAMA_MODEL_INTERVIEWER`, `OLLAMA_MODEL_CRITIC`, and so on, or all of them at once with `OLLAMA_MODEL`.

## Usage

```sh
uv run pytest

uv run python -m interviewer.cli --goal "why people abandon their old note-taking app"

uv run python -m eval.run --personas personas/ --out runs/
uv run python -m eval.metrics runs/
```

## Status

Working end to end against local models. 86 tests cover the parts that need no model at
all: quote validation, the mechanical question rules, routing and probe budgets, the
degeneracy guards, and the full graph driven by a scripted fake client.

What the live runs have shown so far:

- The plumbing works. Structured output via Ollama's JSON-schema `format` parameter
  validates straight into Pydantic models with no parsing layer.
- `qwen3:8b` is not good enough for the critic role. It passed a question that was both
  hypothetical and double-barrelled while writing "this is a compound question" in its own
  feedback field, then later rejected every well-formed question it was shown. Unreliable
  in both directions, which is why the mechanical rules exist and why the critic runs on
  the larger model.
- Every rejected question is now recorded in the saved transcript with its source
  (`rules` or `model`), so a stalled interview can be diagnosed by reading the run instead
  of re-running it under instrumentation.

Not done yet: tuning the interviewer against a model that can sustain a full interview,
and a synthesis run over three real transcripts.
