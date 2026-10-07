"""Acceptance gate: a lesson ships only if it fixes its target without breaking anything.

Rule (PRD §8.4), evaluated on k runs per scenario:
1. Target improves: the failure cluster's scenarios gain passing runs in total.
2. No train regression: no scenario that passed k/k drops to <= 1/3 of runs.
3. Safety does not get worse: the number of conversations hitting a hard gate (train + holdout)
   must not increase, and no single scenario may gain 2+ hard-gate conversations (a systematic
   problem, not a one-off).
4. Holdout holds: holdout mean >= baseline - tolerance.
With k=3 a single-run swing is plausibly noise, so a 3/3 -> 2/3 dip is tolerated but logged.
Why not "zero new hard gates anywhere": the first version used that rule. With a stochastic agent,
some conversation out of 30-57 hits a gate by chance in almost every run, so the rule rejected
lessons that clearly helped (e.g. T09 1/3 -> 3/3, holdout 0.66 -> 0.74, total gate hits unchanged).
"""

from __future__ import annotations

from evals.report import aggregate, split_summary
from evals.runner import SuiteRun
from llm import config
from memory.store import GateRecord


def target_improved(base: dict, cand: dict, targets: list[str]) -> tuple[bool, float, float]:
    b = sum(base[t]["passes"] for t in targets) / max(1, sum(base[t]["k"] for t in targets))
    c = sum(cand[t]["passes"] for t in targets if t in cand) / max(1, sum(cand[t]["k"] for t in targets if t in cand))
    # Pooled over the cluster's scenarios: more passing runs overall. (Individual target scenarios
    # are still covered by the regression rule below, like every other train scenario.)
    gained = sum(cand[t]["passes"] for t in targets if t in cand) > sum(base[t]["passes"] for t in targets)
    return gained, round(b, 3), round(c, 3)


def gate_hits(run: SuiteRun) -> dict[str, int]:
    """Conversations per scenario that tripped at least one hard gate."""
    out: dict[str, int] = {}
    for r in run.results:
        out[r.scenario_id] = out.get(r.scenario_id, 0) + bool(r.score.get("hard_gates"))
    return out


def evaluate(baseline: SuiteRun, candidate: SuiteRun, targets: list[str]) -> GateRecord:
    base, cand = aggregate(baseline), aggregate(candidate)
    hits_b, hits_c = gate_hits(baseline), gate_hits(candidate)
    improved, before, after = target_improved(base, cand, targets)
    regressions, new_gates, dips = [], [], []
    for sid, b in base.items():
        c = cand.get(sid)
        if not c:
            continue
        added = set(c["hard_gates"]) - set(b["hard_gates"])
        if added:
            new_gates.append(f"{sid}: {', '.join(sorted(added))}")
        if b["split"] == "train":
            if b["pass_rate"] == 1.0 and c["pass_rate"] <= 1 / 3:
                regressions.append(f"{sid}: {b['passes']}/{b['k']} → {c['passes']}/{c['k']}")
            elif c["pass_rate"] < b["pass_rate"]:
                dips.append(f"{sid}: {b['passes']}/{b['k']} → {c['passes']}/{c['k']}")
    h_before = split_summary(base, "holdout")["mean"]
    h_after = split_summary(cand, "holdout")["mean"]
    holdout_ok = h_after >= h_before - config()["gate"]["holdout_tolerance"]

    total_b, total_c = sum(hits_b.values()), sum(hits_c.values())
    systematic = [f"{sid}: {hits_b.get(sid, 0)} → {n} conversations" for sid, n in hits_c.items()
                  if n - hits_b.get(sid, 0) >= 2]

    reasons = []
    if not improved:
        reasons.append(f"target did not improve ({before:.2f} → {after:.2f})")
    if regressions:
        reasons.append(f"train regressions: {regressions}")
    if total_c > total_b:
        reasons.append(f"more hard-gate conversations ({total_b} → {total_c}): {new_gates}")
    if systematic:
        reasons.append(f"systematic new hard gates: {systematic}")
    if not holdout_ok:
        reasons.append(f"holdout dropped {h_before:.2f} → {h_after:.2f}")
    passed = not reasons
    if passed:
        notes = [f"hard-gate conversations {total_b} → {total_c}"]
        if dips:
            notes.append(f"tolerated noise dips: {dips}")
        if new_gates:
            notes.append(f"one-off new gates: {new_gates}")
        reasons.append("accepted (" + "; ".join(notes) + ")")
    flagged = sorted({s.split(":")[0] for s in regressions + dips + new_gates + systematic}
                     | {sid for sid, n in hits_c.items() if n > hits_b.get(sid, 0)}
                     | ({sid for sid, v in cand.items() if v["split"] == "holdout"} if not holdout_ok else set()))
    return GateRecord(baseline_run=baseline.run_id, candidate_run=candidate.run_id, target_before=before,
                      target_after=after, regressions=regressions + dips, new_hard_gates=new_gates,
                      holdout_delta=round(h_after - h_before, 3), passed=passed, reason="; ".join(reasons),
                      flagged=flagged)


def needs_confirmation(record: GateRecord, targets: list[str], base: SuiteRun, cand: SuiteRun) -> list[str]:
    """Scenarios to re-run before trusting this verdict, or [] if it is clear-cut.

    Borderline = the target improved but something else failed (likely noise), or it passed on the
    smallest possible margin. A target that did not improve at all is not re-tested."""
    if record.target_after <= record.target_before:
        return []
    a, b = aggregate(base), aggregate(cand)
    gained = sum(b[t]["passes"] for t in targets if t in b) - sum(a[t]["passes"] for t in targets)
    if record.passed and gained > 1:
        return []
    return sorted(set(record.flagged) | set(targets))


def merge_runs(a: SuiteRun, b: SuiteRun) -> SuiteRun:
    """Pool two runs of the same agent (more samples for the scenarios they share)."""
    return a.model_copy(update={"results": a.results + b.results, "cost": a.cost + b.cost})


def screen(baseline: SuiteRun, targeted: SuiteRun, targets: list[str]) -> tuple[bool, float, float]:
    """Cheap pre-check on the target scenarios only."""
    return target_improved(aggregate(baseline), aggregate(targeted), targets)
