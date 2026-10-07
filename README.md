# Self-improving patient scheduling agent

A voice-ready AI receptionist that registers patients and books, reschedules and cancels appointments. It comes with an evaluation harness and a **lessons memory**: failed test calls are turned into general rules, and a rule is kept only if it fixes the failure without breaking anything else.

Built for the 2care.ai take-home. The spec is in [`PRD.md`](PRD.md), the design rationale in [`DESIGN_NOTE.md`](DESIGN_NOTE.md), and where AI helped (or was overruled) in [`AI_USAGE.md`](AI_USAGE.md).

## Setup
```bash
uv sync
cp .env.example .env        # add your OPENAI_API_KEY
uv run pytest               # 34 tests: guardrails, checks, gate, memory, full pipeline on a fake model ($0)
```

## The two commands
```bash
uv run chat                 # talk to the agent as a patient (tool calls shown inline)
uv run improve              # baseline → diagnose failures → propose lessons → screen → gate → before/after report
```

`uv run improve` options: `--k 3` runs per scenario, `--rounds 1`, `--budget-usd 2`, `--approve` (a human signs off on each lesson), `--no-cache` (use this for recorded runs), `--baseline <run_id>` (reuse an existing baseline run).

## Other commands
```bash
uv run evals run --split train|holdout|all --k 1 --only T05,T07   # run scenarios, print the scored table
uv run evals show <run_id> T07_proxy_daughter 0                   # one transcript with its tool calls, checks and hard gates
uv run evals report <run_id> --compare <baseline_run_id>          # before/after table
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
