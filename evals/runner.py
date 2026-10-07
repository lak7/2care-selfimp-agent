"""Run scenarios: simulator <-> agent conversation on a fresh mock EHR, then score."""

from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field
from rich.console import Console

from agent.ehr import MockEHR
from agent.loop import Agent
from agent.prompt import prompt_hash
from evals.asr import ASRNoise
from evals.scenario import Scenario
from evals.simulator import DONE_RE, PatientSimulator
from llm import BudgetExceeded, config

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
_console = Console()


class Result(BaseModel):
    scenario_id: str
    split: str
    run_idx: int
    transcript: list[dict] = Field(default_factory=list)
    trace: list[dict] = Field(default_factory=list)
    before: dict = Field(default_factory=dict)
    after: dict = Field(default_factory=dict)
    ended_by: str = ""
    error: str | None = None
    cost: float = 0.0
    score: dict = Field(default_factory=dict)   # filled by evals.checks.score_result


def run_scenario(sc: Scenario, lessons: list, llm, run_idx: int) -> Result:
    salt = f"{sc.id}#{run_idx}"
    ehr = MockEHR(patch=sc.seed_patch)
    res = Result(scenario_id=sc.id, split=sc.split, run_idx=run_idx, before=ehr.snapshot())
    agent = Agent(ehr, llm, lessons, salt=salt)
    sim = PatientSimulator(sc, llm, salt)
    asr = ASRNoise(salt, sc.behaviors.asr_confusions) if sc.behaviors.asr_noise else None
    llm.start_tracking()
    script = dict(sc.persona.script)
    reactions = list(sc.persona.reactions)
    said = sc.persona.opening_line
    pending_events = list(sc.world_events)

    def agent_turn(turn: int, said: str) -> str:
        heard = asr(said) if asr else said
        res.transcript.append({"role": "patient", "turn": turn, "said": said, "text": heard})
        for ev in list(pending_events):
            if ev.action == "take_held_slot" and turn > ev.after_turn and agent.session.held_slot_id:
                ehr.external_book(agent.session.held_slot_id)
                res.trace.append({"turn": turn, "tool": None, "event": "world:take_held_slot",
                                  "slot_id": agent.session.held_slot_id})
                pending_events.remove(ev)
        reply = agent.step(heard)
        res.transcript.append({"role": "agent", "turn": turn, "text": reply})
        if agent.session.emergency_locked and _mentions_911(reply):
            res.ended_by = "emergency"
        return reply

    try:
        for turn in range(1, sc.max_turns + 1):
            agent_turn(turn, said)
            nxt = turn + 1
            if nxt in script:                      # scripted event: always said, verbatim
                said = script.pop(nxt)
                continue
            last = res.transcript[-1]["text"]
            hit = next((r for r in reactions if re.search(r["when"], last, re.I)), None)
            if hit:                                # deterministic correction of a wrong read-back
                reactions.remove(hit)
                said = hit["say"]
                continue
            said = sim.reply(res.transcript, turn)
            if DONE_RE.search(said):
                if script:                         # don't let the patient hang up before the event
                    said = script.pop(min(script))
                    continue
                tail = DONE_RE.sub("", said).strip()
                if tail and turn < sc.max_turns:   # e.g. "Yes, please cancel it. [DONE]" -> agent must act
                    agent_turn(nxt, tail)
                res.ended_by = res.ended_by or "done"
                break
        else:
            res.ended_by = "max_turns"
    except BudgetExceeded as e:
        res.error, res.ended_by = f"budget: {e}", "budget"
    except Exception as e:  # keep the suite running; a crash is scored as a failure
        res.error, res.ended_by = f"{type(e).__name__}: {e}", "error"
    res.trace = sorted(agent.trace + res.trace, key=lambda t: t["turn"])
    res.after = ehr.snapshot()
    res.cost = llm.tracked()
    return res


def _mentions_911(text: str) -> bool:
    return "911" in text or "emergency" in text.lower()


class SuiteRun(BaseModel):
    run_id: str
    created_at: str
    split: str
    k: int
    prompt_hash: str
    lesson_ids: list[str]
    results: list[Result]
    cost: float = 0.0
    label: str = ""


def new_run_id(label: str = "") -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S") + (f"-{label}" if label else "")


def run_suite(scenarios: list[Scenario], lessons: list, llm, k: int, split: str = "all",
              label: str = "", run_id: str | None = None, progress: bool = True,
              idx_offset: int = 0) -> SuiteRun:
    """All conversations run concurrently; each is judged as soon as it finishes."""
    from evals.checks import score_result

    run_id = run_id or new_run_id(label)
    jobs = [(sc, i) for sc in scenarios for i in range(idx_offset, idx_offset + k)]
    done = [0]
    lock = threading.Lock()

    def one(job) -> Result:
        sc, i = job
        r = run_scenario(sc, lessons, llm, i)
        llm.start_tracking()
        r.score = score_result(sc, r, llm)
        r.score["judge_cost"] = llm.tracked()
        if progress:
            with lock:
                done[0] += 1
                gates = ",".join(g.split("_")[0] for g in r.score.get("hard_gates", {}))
                mark = "[green]✓[/]" if r.score.get("passed") else "[red]✗[/]"
                _console.print(f"  {mark} {done[0]:>3}/{len(jobs)}  {sc.id}#{i:<2} score {r.score.get('total', 0):.2f}"
                               f"  {('[red]' + gates + '[/]') if gates else ''}  [dim]${llm.spent:.3f}[/]")
        return r

    with ThreadPoolExecutor(max_workers=min(config()["eval"]["concurrency"], len(jobs) or 1)) as pool:
        results = list(pool.map(one, jobs))
    run = SuiteRun(run_id=run_id, created_at=datetime.now().isoformat(timespec="seconds"),
                   split=split, k=k, prompt_hash=prompt_hash(MockEHR(), lessons),
                   lesson_ids=[l.id for l in lessons], results=results,
                   cost=sum(r.cost + r.score.get("judge_cost", 0) for r in results), label=label)
    save_run(run)
    return run


def save_run(run: SuiteRun) -> Path:
    d = RUNS_DIR / run.run_id
    (d / "transcripts").mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(run.model_dump_json(indent=1))
    for r in run.results:
        (d / "transcripts" / f"{r.scenario_id}#{r.run_idx}.json").write_text(json.dumps(
            {"transcript": r.transcript, "trace": r.trace, "score": r.score, "ended_by": r.ended_by,
             "error": r.error}, indent=1, default=str))
    return d


def load_run(run_id: str) -> SuiteRun:
    return SuiteRun.model_validate_json((RUNS_DIR / run_id / "results.json").read_text())


def latest_matching_run(scenario_ids: set[str], lessons: list, k: int) -> SuiteRun | None:
    """Most recent saved run usable as a baseline for the current agent: same prompt + lessons
    (prompt hash), same k, covering every scenario, and not cut short by the budget."""
    want = prompt_hash(MockEHR(), lessons)
    for d in sorted((p for p in RUNS_DIR.iterdir() if (p / "results.json").exists()), reverse=True):
        try:
            run = load_run(d.name)
        except Exception:
            continue
        if run.prompt_hash != want or run.k != k:
            continue
        if {r.scenario_id for r in run.results} != scenario_ids:
            continue
        if any(r.error and r.error.startswith("budget") for r in run.results):
            continue
        return run
    return None
