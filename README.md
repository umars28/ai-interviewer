# ai-interviewer

An agent that **interviews people** instead of answering them.

Give it a research goal. It talks to a respondent over chat, decides when to probe deeper and when to move on, refuses to ask leading questions, and stops when the research goals are covered — not when the model feels done. After several interviews it synthesises themes across respondents, and every theme must carry a quote that provably exists in a transcript.

## Why this exists

Most LLM projects cannot be evaluated: the only judge is "looks good to me". This one can. Synthetic respondents are seeded with facts that only surface under correct probing, which turns interview quality into four numbers:

| Metric | How it is computed |
| --- | --- |
| Hidden fact recall | Of 5 planted facts, how many the agent surfaced |
| Leading question rate | Questions that violated the interview rules but still reached the respondent |
| Turns to coverage | Turns needed before every research goal is covered |
| Quote validity | Must be 100% — checked by string matching, not by a model |

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

Work in progress.
