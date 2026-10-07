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
from evals.judge import trace_summary
from evals.runner import Result, SuiteRun
from evals.scenario import Scenario
from memory.store import Lesson, LessonSource


class LessonProposal(BaseModel):
    action: Literal["new", "merge"]
    merge_into: str | None
    target: Literal["playbook", "tool_desc", "needs_code"]
    tool: str | None
    trigger: str
    rule: str
    root_cause: str
    evidence: str


OPTIMIZER_INSTRUCTIONS = """You improve an AI phone receptionist for a medical clinic by writing ONE lesson
that fixes a cluster of failed test calls.

A lesson is a general behavioural rule: "When <trigger>: <rule>". Requirements:
- General: it must help on unseen calls of the same kind. NEVER mention specific people, names, dates,
  times, scenario ids or test details.
- Short and actionable (one or two sentences), phrased as an instruction to the receptionist.
- Choose the target:
  - "playbook": a conversational/procedural rule for the system prompt (most cases).
  - "tool_desc": a usage note that belongs on one specific tool's description (set `tool`).
  - "needs_code": the failure can only be prevented reliably by a code guardrail or a tool change
    (e.g. the model was ABLE to do something unsafe, or a tool returns wrong data). Describe the
    guardrail in `rule`; a human will implement it. Do not use prompt text to patch a code problem.
- If an existing lesson already covers this failure but is not working, use action "merge" with
  `merge_into` set to its id and write an improved rule that replaces it. Do not duplicate lessons.
- Do not re-propose a rule that was already rejected unless it is materially different.
- `evidence` quotes the specific turn(s) that show the failure; `root_cause` explains why it happened."""


def failure_ids(r: Result) -> list[str]:
    ids = list(r.score.get("hard_gates", {}))
    ids += [c["id"] for c in r.score.get("checks", [])
            if not c["passed"] and (c["layer"] != "judge" or c["detail"].startswith("0/"))]
    if r.error:
        ids.append("error")
    return ids


def cluster_failures(run: SuiteRun, max_clusters: int = 3) -> list[dict]:
    """Group failed train results by their primary violation (hard gate first, then the first check)."""
    clusters: dict[str, list[Result]] = defaultdict(list)
    for r in run.results:
        if r.split != "train" or r.score.get("passed") or r.error:
            continue
        ids = failure_ids(r)
        if ids:
            clusters[ids[0]].append(r)
    ranked = sorted(clusters.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:max_clusters]
    return [{"key": k, "results": v, "scenarios": sorted({r.scenario_id for r in v})} for k, v in ranked]


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


def lint(p: LessonProposal, scenarios: list[Scenario]) -> list[str]:
    """Anti-overfit lint enforced in code: lessons must not reference test specifics."""
    text = f"{p.trigger} {p.rule}".lower()
    problems = [f"mentions '{t}'" for t in _forbidden_terms(scenarios)
                if re.search(rf"\b{re.escape(t)}\b", text) and t not in {"care", "primary", "dental"}]
    if re.search(MONTHS + r"\s+\d", text) or re.search(r"\b[th]\d{2}_", text):
        problems.append("mentions a specific date or scenario id")
    if len(p.rule) > 450:
        problems.append("rule is too long (max ~450 chars)")
    if p.target == "tool_desc" and p.tool not in TOOLS:
        problems.append(f"unknown tool {p.tool!r}")
    return problems


def _example(r: Result) -> str:
    convo = "\n".join(f"{'CALLER' if m['role'] == 'patient' else 'RECEPTIONIST'} (t{m['turn']}): {m['text']}"
                      for m in r.transcript)
    fails = [f"- {c['id']}: {c['detail']}" for c in r.score.get("checks", []) if not c["passed"]]
    gates = [f"- HARD GATE {g}: {ev}" for g, ev in r.score.get("hard_gates", {}).items()]
    return (f"### Failed call ({r.scenario_id} run {r.run_idx}, score {r.score.get('total')})\n"
            f"Violations:\n" + "\n".join(gates + fails) +
            f"\n\nTranscript:\n{convo}\n\nTool calls:\n{trace_summary(r.trace)}")


def propose(cluster: dict, system_prompt: str, tool_descriptions: dict[str, str], lessons: list[Lesson],
            scenarios: list[Scenario], llm, store_next_id: str) -> Lesson | None:
    examples, seen = [], set()
    for r in sorted(cluster["results"], key=lambda r: r.score.get("total", 0)):
        if r.scenario_id not in seen and len(examples) < 2:
            examples.append(r)
            seen.add(r.scenario_id)
    existing = [{"id": l.id, "status": l.status, "target": l.target, "trigger": l.trigger, "rule": l.rule}
                for l in lessons]
    prompt = (f"## Current system prompt\n{system_prompt}\n\n## Tool descriptions\n"
              f"{json.dumps(tool_descriptions, indent=1)}\n\n## Existing lessons\n"
              f"{json.dumps(existing, indent=1) if existing else '(none)'}\n\n"
              f"## Failure cluster: {cluster['key']} (in {len(cluster['results'])} run(s) across "
              f"{', '.join(cluster['scenarios'])})\n\n" + "\n\n".join(_example(r) for r in examples))
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
                trigger=p.trigger, rule=p.rule,
                merged_from=[p.merge_into] if p.action == "merge" and p.merge_into else [])
            return lesson
        feedback = ("\n\n## Your previous proposal was rejected by the lint\n" + "\n".join(problems) +
                    f"\nPrevious rule: {p.rule}\nRewrite it as a general rule.")
    return None
