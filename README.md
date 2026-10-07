# Self-improving patient scheduling agent

A voice-ready AI receptionist that registers patients and books, reschedules and cancels appointments. It comes with an evaluation harness and a **lessons memory**: failed test calls are turned into general rules, and a rule is kept only if it fixes the failure without breaking anything else.

Built for the 2care.ai take-home. The spec is in [`PRD.md`](PRD.md), the design rationale in [`DESIGN_NOTE.md`](DESIGN_NOTE.md), and where AI helped (or was overruled) in [`AI_USAGE.md`](AI_USAGE.md).

## Setup
```bash
uv sync
cp .env.example .env        # add your OPENAI_API_KEY
uv run pytest               # 40 tests: guardrails, checks, gate, memory, full pipeline on a fake model ($0)
```

## The two commands
```bash
uv run chat                 # talk to the agent as a patient (tool calls shown inline)
uv run improve              # diagnose the latest eval run → propose lessons → screen → gate → before/after report
```

## The flow
```bash
uv run evals run --split all --k 3     # 1. measure + record every conversation, check and failure in runs/<id>/
uv run evals show <id> T07_proxy_daughter 0   #    inspect any failure
uv run improve                          # 2. learn from that run (re-runs only to screen and gate each lesson)
uv run memory list                      # 3. what was accepted / rejected / flagged for a human, and why
```
By default, `improve` reuses the **most recent saved run whose prompt, lessons and k match the current agent and that covers all scenarios**. If there isn't one, it runs the baseline itself. `improve` still has to run the agent to *screen* and *gate* each lesson: proving that a lesson helps and breaks nothing requires new conversations with it in the prompt.

`uv run improve` options:
- `--fresh-baseline`: rerun the baseline even if a matching run exists.
- `--baseline <run_id>`: use a specific run as the baseline.
- `--k 3`: runs per scenario. Must match the baseline run.
- `--rounds 1`
- `--budget-usd 2`
- `--approve`: a human signs off on each lesson.
- `--no-cache`: use this for recorded runs.

## Profiles: full vs demo
- `--profile full` (the default): all 19 scenarios at k=3. This is the real result. A full suite takes about 1 minute.
- `--profile demo`: 10 scenarios at k=2, with one lesson per round so screening is skipped. It's the same loop sized for a 5-minute recording: eval about 40 s, improve about 75 s. Its statistics are weaker.

The scenario lists are in `config.yaml` under `profiles`.
```bash
uv run evals run --profile demo && uv run improve --profile demo && uv run evals run --profile demo
```

## Other commands
```bash
uv run evals run --split train|holdout|all --k 1 --only T05,T07   # run scenarios, print the scored table
uv run evals show <run_id> T07_proxy_daughter 0                   # one transcript with its tool calls, checks and hard gates
uv run evals report <run_id> --compare <baseline_run_id>          # before/after table
uv run evals rescore <run_id>                                      # re-apply the current checks to a saved run (judge cached, ~$0)
uv run evals cost                                                  # total OpenAI spend so far, by role
uv run memory list | show L-001 | retire L-001 | restore L-001 | ablate L-001 | reset --yes
uv run chat --no-lessons                                           # chat with the v0 agent
```

## Layout
```
agent/    ehr.py (EHRAdapter + MockEHR) · session.py · tools.py (guardrails) · prompt.py · loop.py · cli.py
evals/    scenarios/{train,holdout} · simulator.py · asr.py · checks.py · judge.py · runner.py · report.py
improve/  diagnose.py (failures → lesson) · gate.py (accept rule) · loop.py
memory/   lessons.json · store.py · compile.py (lessons → Playbook + tool notes)
llm.py    the only module that calls OpenAI: cost ledger, budget caps, dev cache
runs/     every run: results.json, transcripts/, report.md; ledger.jsonl (total spend)
```

## Cost
- Agent and patient simulator run on `gpt-5-nano`; the judge and optimizer run on `gpt-5-mini`.
- A full suite (19 scenarios × k=3) costs about $0.5. One improvement loop costs about $1–1.5.
- Every call is recorded in `runs/ledger.jsonl`. Calls are refused once the total reaches $6.50 (cap $7 minus a margin).
- Repeating an identical request is free while developing, because responses are cached in `.cache/`. Use `--no-cache` to turn this off.
