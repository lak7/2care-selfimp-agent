"""Turn failed train runs into structured lesson proposals.

Only TRAIN results are ever shown to the optimizer; holdout stays unseen so it can
measure whether lessons generalise.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Literal

from pydantic import BaseModel

from agent.ehr import MockEHR
from agent.tools import TOOLS
from evals.checks import EXPECTED, STYLE_CHECKS
from evals.judge import GLOBAL_ITEMS, trace_summary
from evals.runner import Result, SuiteRun
from evals.scenario import Scenario
from memory.compile import clean_trigger
from memory.store import Lesson, LessonSource


class LessonProposal(BaseModel):
    action: Literal["new", "merge"]
    merge_into: str | None
    target: Literal["playbook", "tool_desc", "needs_code"]
    tool: str | None
    trigger: str
    rule: str
    example: str | None
    root_cause: str
    evidence: str


OPTIMIZER_INSTRUCTIONS = """You improve an AI phone receptionist for a medical clinic by writing ONE lesson
that fixes a cluster of failed test calls.

A lesson is a general behavioural rule: "When <trigger>: <rule>". Requirements:
- General: it must help on unseen calls of the same kind. NEVER mention specific people, names, dates,
  times, scenario ids or test details.
- Short and actionable: at most two sentences and 400 characters, phrased as an instruction to the receptionist.
- Choose the target:
  - "playbook": a conversational/procedural rule for the system prompt (most cases).
  - "tool_desc": a usage note that belongs on one specific tool's description (set `tool`).
  - "needs_code": the failure can only be prevented reliably by a code guardrail or a tool change
    (e.g. the model was ABLE to do something unsafe, or a tool returns wrong data). Describe the
    guardrail in `rule`; a human will implement it. Do not use prompt text to patch a code problem.
- If an existing lesson already covers this failure but is not working, use action "merge" with
  `merge_into` set to its id and write an improved rule that replaces it. Do not duplicate lessons.
- Do not re-propose a rule that was already rejected unless it is materially different.
- `evidence` quotes the specific turn(s) that show the failure; `root_cause` explains why it happened.

What actually changes this agent's behaviour (it is a small model):
- Do NOT restate an instruction the system prompt already contains. If the prompt already says it and the
  agent still violates it, repeating it will not help: write a narrower, step-by-step procedure tied to
  the exact moment it applies (e.g. "before calling cancel_appointment, ..."), prefer a tool_desc note on
  the tool used at that moment, or choose needs_code if only code can guarantee it (e.g. output formatting
  for text-to-speech).
- Give a short `example` of what the receptionist should actually say in that moment (one sentence).
  Use null only when an example makes no sense."""


def failure_ids(r: Result) -> list[str]:
    ids = list(r.score.get("hard_gates", {}))
    ids += [c["id"] for c in r.score.get("checks", [])
            if not c["passed"] and (c["layer"] != "judge" or c["detail"].startswith("0/"))]
    if r.error:
        ids.append("error")
    return ids


def cluster_failures(run: SuiteRun, max_clusters: int = 3) -> list[dict]:
    """Rank violations across failed TRAIN runs: hard gates first (safety), then by how many
    failed runs show the violation. Each cluster = every failed run that has that violation."""
    clusters: dict[str, list[Result]] = defaultdict(list)
    for r in run.results:
        if r.split != "train" or r.score.get("passed") or r.error:
            continue
        for vid in dict.fromkeys(failure_ids(r)):
            if vid not in STYLE_CHECKS:      # style is routed to a code fix, not the prompt
                clusters[vid].append(r)
    ranked = sorted(clusters.items(), key=lambda kv: (not kv[0].startswith("HG"), -len(kv[1]), kv[0]))
    return [{"key": k, "results": v, "scenarios": sorted({r.scenario_id for r in v})}
            for k, v in ranked[:max_clusters]]


MONTH_NAMES = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
               "october", "november", "december"]


def _test_dates(scenarios: list[Scenario]) -> set[str]:
    """'march 14'-style dates that come from the test data (patient DOBs, seeded appointments,
    persona facts). Illustrative dates in a rule are fine; these would be memorising the test."""
    ehr = MockEHR()
    dates = {f"{MONTH_NAMES[p.dob.month - 1]} {p.dob.day}" for p in ehr.patients.values()}
    dates |= {f"{MONTH_NAMES[ehr.slots[a.slot_id].start.month - 1]} {ehr.slots[a.slot_id].start.day}"
              for a in ehr.appointments.values()}
    for sc in scenarios:
        for v in sc.persona.hidden_facts.values():
            dates |= {f"{m.group(1).lower()} {int(m.group(2))}"
                      for m in re.finditer(MONTHS + r"\s+(\d{1,2})", v, re.I)}
    return dates


def _forbidden_terms(scenarios: list[Scenario]) -> set[str]:
    terms = set()
    for p in MockEHR().patients.values():
        terms |= {w.lower() for w in p.full_name.split() if len(w) > 2}
    for sc in scenarios:
        terms.add(sc.id.lower())
        for k, v in sc.persona.hidden_facts.items():
            if "name" in k:
                terms |= {w.lower() for w in v.split() if len(w) > 2}
    return terms


MONTHS = r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b"


# Triggers that fire on every turn give a rule side effects everywhere (in testing, "keep every reply
# short" made the agent drop emergency handling). Lessons must name a specific moment.
GLOBAL_TRIGGER = re.compile(r"\b(any|every|each|all)\s+(spoken\s+)?(reply|replies|turn|turns|response|responses|"
                            r"message|messages|time|call|calls|interaction|conversation)\b|\balways\b|"
                            r"\b(speaking|replying|responding)\b|\bspoken (reply|turn|response)\b|\bspeak a\b")


def lint(p: LessonProposal, scenarios: list[Scenario]) -> list[str]:
    """Anti-overfit lint enforced in code: lessons must not reference test specifics."""
    text = f"{p.trigger} {p.rule}".lower()
    problems = [f"mentions '{t}'" for t in _forbidden_terms(scenarios)
                if re.search(rf"\b{re.escape(t)}\b", text) and t not in {"care", "primary", "dental"}]
    for m in re.finditer(MONTHS + r"\s+(\d{1,2})", text):
        if f"{m.group(1)} {int(m.group(2))}" in _test_dates(scenarios):
            problems.append(f"mentions '{m.group(0)}', a date from the test data")
    if re.search(r"\b[th]\d{2}_", text):
        problems.append("mentions a scenario id")
    if p.target != "needs_code" and GLOBAL_TRIGGER.search(p.trigger.lower()):
        problems.append(f"trigger '{p.trigger}' applies to every turn; name the specific moment it applies "
                        "(e.g. 'before calling cancel_appointment', 'when the caller gives a numeric date')")
    if len(p.rule) > 500:
        problems.append(f"rule is too long ({len(p.rule)} chars, max 400): keep only the essential step")
    if p.target == "tool_desc" and p.tool not in TOOLS:
        problems.append(f"unknown tool {p.tool!r}")
    return problems


def expected_for(check_id: str, sc: Scenario | None) -> str:
    if check_id.startswith("judge:"):
        items = {i.id: i.text for i in GLOBAL_ITEMS + (sc.expect.judge if sc else [])}
        return items.get(check_id.split(":", 1)[1], "")
    return EXPECTED.get(check_id, "")


def _convo(r: Result) -> str:
    return "\n".join(f"{'CALLER' if m['role'] == 'patient' else 'RECEPTIONIST'} (t{m['turn']}): {m['text']}"
                     for m in r.transcript)


def _example(r: Result, key: str, sc: Scenario | None, good: Result | None) -> str:
    focus = [f"- HARD GATE {g}: {ev}" for g, ev in r.score.get("hard_gates", {}).items() if g == key]
    focus += [f"- {c['id']}: {c['detail']}" for c in r.score.get("checks", []) if c["id"] == key and not c["passed"]]
    out = (f"### Failed call ({r.scenario_id} run {r.run_idx}: {sc.title if sc else ''})\n"
           f"What went wrong:\n" + "\n".join(focus) +
           f"\nWhat was expected: {expected_for(key, sc)}\n\n"
           f"Transcript:\n{_convo(r)}\n\nTool calls:\n{trace_summary(r.trace)}")
    if good:
        out += (f"\n\n### For contrast: a run of the SAME scenario that got this right\n{_convo(good)}\n"
                f"Tool calls:\n{trace_summary(good.trace)}")
    return out


def propose(cluster: dict, system_prompt: str, tool_descriptions: dict[str, str], lessons: list[Lesson],
            scenarios: list[Scenario], llm, store_next_id: str,
            baseline: SuiteRun | None = None) -> tuple[Lesson | None, list[str]]:
    examples, seen = [], set()
    for r in sorted(cluster["results"], key=lambda r: r.score.get("total", 0)):
        if r.scenario_id not in seen and len(examples) < 2:
            examples.append(r)
            seen.add(r.scenario_id)
    existing = [{"id": l.id, "status": l.status, "target": l.target, "trigger": l.trigger, "rule": l.rule,
                 "gate": l.gate.reason if l.gate else None} for l in lessons]
    by_id = {s.id: s for s in scenarios}
    key = cluster["key"]

    def passing_run(sid: str) -> Result | None:
        if not baseline:
            return None
        ok = [r for r in baseline.results if r.scenario_id == sid and not r.error
              and key not in failure_ids(r)]
        return max(ok, key=lambda r: r.score.get("total", 0)) if ok else None

    guards = sorted({r.scenario_id for r in (baseline.results if baseline else [])
                     if r.split == "train" and r.score.get("passed")} - set(cluster["scenarios"]))
    guard_txt = "\n".join(f"- {sid}: {by_id[sid].title}" for sid in guards if sid in by_id)
    prompt = (f"## Current system prompt\n{system_prompt}\n\n## Tool descriptions\n"
              f"{json.dumps(tool_descriptions, indent=1)}\n\n## Existing lessons\n"
              f"{json.dumps(existing, indent=1) if existing else '(none)'}\n\n"
              f"## Failure cluster: {cluster['key']} (in {len(cluster['results'])} run(s) across "
              f"{', '.join(cluster['scenarios'])})\n"
              f"Expected behaviour: {expected_for(key, by_id.get(examples[0].scenario_id))}\n"
              f"Write ONE lesson that fixes `{key}`. The examples may show other violations too; ignore "
              f"those, they are handled separately.\n\n"
              + "\n\n".join(_example(r, key, by_id.get(r.scenario_id), passing_run(r.scenario_id))
                              for r in examples)
              + (f"\n\n## Scenarios that currently pass. Your rule must not change behaviour in them:\n{guard_txt}"
                 if guard_txt else ""))
    feedback = ""
    for attempt in range(2):
        res = llm.respond("optimizer", [{"role": "user", "content": prompt + feedback}],
                          instructions=OPTIMIZER_INSTRUCTIONS, text_format=LessonProposal,
                          salt=f"{cluster['key']}:{attempt}:{len(lessons)}")
        p: LessonProposal = res.parsed
        problems = lint(p, scenarios)
        if not problems:
            target = f"tool_desc:{p.tool}" if p.target == "tool_desc" else p.target
            first = examples[0]
            lesson = Lesson(id=store_next_id, source=LessonSource(
                scenario=first.scenario_id, run_id="", transcript=f"{first.scenario_id}#{first.run_idx}"),
                violated=sorted({i for r in cluster["results"] for i in failure_ids(r)}),
                evidence=p.evidence, root_cause=p.root_cause, target=target,
                trigger=clean_trigger(p.trigger), rule=p.rule, example=p.example, cluster=cluster["key"],
                merged_from=[p.merge_into] if p.action == "merge" and p.merge_into else [])
            return lesson, []
        feedback = ("\n\n## Your previous proposal was rejected by the lint\n" + "\n".join(problems) +
                    f"\nPrevious rule: {p.rule}\nRewrite it as a general rule.")
    return None, problems
