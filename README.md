# ai-interviewer

An agent that **interviews people** instead of answering them.

Give it a research goal. It talks to a respondent, decides when to probe deeper and when to
move on, refuses to ask leading questions, and stops when the research goals are covered —
not when the model feels done. Across several interviews it synthesises themes, and every
theme must carry a quote that provably exists in a transcript.

Real output from `python -m eval.run`, no editing:

```
Q: Can you describe the last time you tried to use the note-taking app and
   something went wrong?
A: The last time I tried to use the note-taking app, it got slow. I was in the
   middle of an incident, and opening a note took fourteen seconds. Three people
   were waiting on a call, and I couldn't even get a line in.

Q: What specific features or functions did you find were missing or not working
   well in the app during that time?
A: It didn't handle the volume. The app was built for small teams, not for someone
   who uses it as a log for incidents and code reviews. The search was too slow,
   and the file just kept growing.
```

What the respondent never saw is the other half of the run. Every question is checked before
it is sent, and the saved transcript records each rejection with its cause — which is how
the false positive below was found rather than guessed at.

| | |
| --- | --- |
| Interviews | 3 synthetic respondents, stopping on goal coverage rather than a turn cap |
| Planted-fact recall | 0.29 mean over nine runs — statistically indistinguishable from a one-prompt control at 0.27 |
| Repeated or rule-breaking questions sent | 0 |
| Quote validity in synthesis | 92% — one near-miss paraphrase caught and dropped |

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

Working end to end against local models. 91 tests cover the parts that need no model at
all: quote validation, the mechanical question rules, routing and probe budgets, the
evidence invariant, the degeneracy guards, and the full graph driven by a scripted fake
client.

Baseline over nine interviews — three personas, three repeats each — on `qwen3:14b` /
`qwen3:8b`:

| persona | mean | min | max |
| --- | --- | --- | --- |
| devan | 0.20 | 0.20 | 0.20 |
| hanna | 0.27 | 0.20 | 0.40 |
| mira | 0.40 | 0.20 | 0.60 |
| **overall** | **0.29** | 0.20 | 0.60 |

All nine were valid: every one stopped on goal coverage rather than the turn cap, none
repeated a question, none let a rule-breaking question through, and one forced fallback
occurred across all nine runs. Between 21 and 31 facts were extracted per interview.

### The control: does any of this machinery help?

A naive interviewer — one prompt, no graph, no critic, no coverage tracking, no probe
budget — was run on the same three personas, three repeats each, with the same interviewer
model, the same simulated respondents, the same six-turn budget, and the same recall
metric. Only the interviewer side differs.

| | mean | median | sd | min | runs scoring 0.00 |
| --- | --- | --- | --- | --- | --- |
| This project | 0.29 | 0.20 | 0.15 | 0.20 | 0 of 9 |
| Naive, one prompt | 0.27 | 0.20 | 0.25 | 0.00 | 3 of 9 |

**On the headline metric the graph buys nothing.** A difference of 0.02 between means whose
standard deviations are 0.15 and 0.25 is not a result. Identical medians. Anyone reading
this repo should know that before reading the architecture section.

One difference does survive: the naive interviewer returned nothing at all on three of nine
runs, and this one never dropped below 0.20. Same average, half the variance, no total
failures. That is consistent with the coverage tracking and probe budget acting as a floor
rather than a ceiling — but n=9 makes it suggestive, not established.

Two honest caveats. The personas and the recall metric were both written by the same author
as the system under test, so the metric may simply not capture what the extra machinery is
good at. And the guarantees this project does provide — no leading or repeated question
reaching a respondent, a stop condition tied to goal coverage rather than a fixed turn
count, synthesis quotes verified against transcripts — are things the naive control does not
attempt at all, so recall was never going to measure them.

### Why the repeats exist

Two earlier single passes over the same personas reported mean recall of **0.47** and
**0.10**. Nothing between them should have hurt recall — the only interview-facing change
loosened a rule that had been rejecting valid questions — yet the number fell by a factor
of four.

The nine-run baseline lands at 0.29, between the two. Both earlier figures were sampling
noise around roughly the same underlying quality, and neither meant what it appeared to
mean. An earlier version of this README claimed a specific improvement on the strength of
one of them. That claim was unsupported.

So `eval.run` takes `--repeat N`, reports mean, min and max per persona, and prints an
explicit warning when the range is wide enough that single-run comparisons cannot show
anything:

```sh
uv run python -m eval.run --repeat 3 --out runs/
```

The per-persona spread also localises the difficulty. Devan returns 0.20 on every run —
his planted facts resist the current probing strategy consistently, which is a solvable
problem. Mira ranges 0.20 to 0.60, so her facts are reachable and the interviewer
sometimes fails to reach them, which is a different problem. One number would have hidden
both.

What the live runs have shown so far:

- The plumbing works. Structured output via Ollama's JSON-schema `format` parameter
  validates straight into Pydantic models with no parsing layer.
- `qwen3:8b` is not good enough for the critic role. It passed a question that was both
  hypothetical and double-barrelled while writing "this is a compound question" in its own
  feedback field, then later rejected every well-formed question it was shown. Unreliable
  in both directions, which is why the mechanical rules exist and why the critic runs on
  the larger model.
- **A mechanical rule is only as good as its false-positive rate.** Two phrasings that
  read as violations to a regex are standard research practice: `Can you tell me about…`
  is an invitation to tell a story, not a yes/no question, and `How would you describe…`
  asks for an existing view, not a hypothetical one. Both were rejected three times in a
  row and forced the agent onto a generic fallback. Both were found by reading the
  rejection log after a run, not by reasoning about the regex.
- **Schema breadth suppresses list extraction under constrained decoding.** The assessor
  returned `facts: []` on answers full of facts. Same model, same prompt, same answer: a
  four-field schema (`facts`, `kind`, `goal_progress`, `emergent_topic`) extracted nothing,
  while a one-field `{facts: [...]}` schema extracted three correct facts. Prompting did
  not move it; splitting extraction and classification into two calls took the count from
  0 to 16 in one interview. Worth knowing before blaming a model for ignoring a prompt.
- The assessor also marked goals covered while extracting no facts, so an interview could
  stop on coverage with nothing to read back. The fix that holds is the invariant in
  `assess.py` — a goal cannot reach `covered` without at least one extracted fact, and an
  answer called concrete with no facts is reclassified as vague and probed again.
- Every rejected question is now recorded in the saved transcript with its source
  (`rules` or `model`), so a stalled interview can be diagnosed by reading the run instead
  of re-running it under instrumentation.

### The quote validator earning its keep

`docs/example-synthesis.md` is a real synthesis over those three transcripts. It reports
**92% quote validity** — four themes and two contradictions survived, and one quote was
dropped. The dropped one is the interesting part:

| | |
| --- | --- |
| Model wrote | "Last time that came up? It got slow. I was in the middle of an incident…" |
| Devan actually said | "The last time I tried to use the note-taking app, it got slow. I was in the middle of an incident…" |

Everything after the first clause matches. The opening was rewritten into something more
quotable. No reviewer skimming a findings deck would catch that, and no LLM judge reliably
would either — it is a faithful paraphrase of a real statement. A string comparison caught
it and threw the theme's evidence out.

That is the whole argument for doing verification in code rather than asking a model to
check itself.

The open question is not "how do we raise 0.29" but "does the graph earn its keep at all".
The control says it does not, on this metric. Three ways to settle that, in the order worth
trying:

1. **Raise n.** Nine runs per arm cannot separate 0.29 from 0.27. Twenty per arm could, and
   would also test whether the zero-failure floor is real or luck.
2. **Measure what the graph actually claims.** Recall was the wrong instrument for testing
   coverage-driven stopping and question quality. Score whether each research goal was
   genuinely addressed, and have a judge blind to the source rate transcript quality.
3. **Fix the interviewer, then re-run both arms.** It has never been tuned. A rejected
   question gets reworded rather than fixed — shown a double-barrelled question three times
   in a row, it rephrased it into another double-barrelled question each time instead of
   dropping one half.
