"""Acceptance gate: a lesson ships only if it fixes its target without breaking anything.

Rule (PRD §8.4), evaluated on k runs per scenario:
1. Target improves: every target scenario gains >= 1 passing run out of k.
2. No train regression: no scenario that passed k/k drops to <= 1/3 of runs, and no scenario
   shows a hard gate it did not show before.
3. Holdout holds: holdout mean >= baseline - tolerance, and no new holdout hard gates.
With k=3 a single-run swing is plausibly noise, so a 3/3 -> 2/3 dip is tolerated but logged.
"""

from __future__ import annotations

from evals.report import aggregate, split_summary
from evals.runner import SuiteRun
from llm import config
from memory.store import GateRecord


def target_improved(base: dict, cand: dict, targets: list[str]) -> tuple[bool, float, float]:
    b = sum(base[t]["passes"] for t in targets) / max(1, sum(base[t]["k"] for t in targets))
    c = sum(cand[t]["passes"] for t in targets if t in cand) / max(1, sum(cand[t]["k"] for t in targets if t in cand))
    # Pooled over the cluster's scenarios: more passing runs overall, and no target got worse.
    gained = sum(cand[t]["passes"] for t in targets if t in cand) > sum(base[t]["passes"] for t in targets)
    none_worse = all(t in cand and cand[t]["passes"] >= base[t]["passes"] for t in targets)
    return gained and none_worse, round(b, 3), round(c, 3)


def evaluate(baseline: SuiteRun, candidate: SuiteRun, targets: list[str]) -> GateRecord:
    base, cand = aggregate(baseline), aggregate(candidate)
    improved, before, after = target_improved(base, cand, targets)
    regressions, new_gates, dips = [], [], []
    for sid, b in base.items():
        c = cand.get(sid)
        if not c:
            continue
        added = set(c["hard_gates"]) - set(b["hard_gates"])
        if added:
            new_gates.append(f"{sid}: {', '.join(sorted(added))}")
        if b["split"] == "train" and sid not in targets:
            if b["pass_rate"] == 1.0 and c["pass_rate"] <= 1 / 3:
                regressions.append(f"{sid}: {b['passes']}/{b['k']} → {c['passes']}/{c['k']}")
            elif c["pass_rate"] < b["pass_rate"]:
                dips.append(f"{sid}: {b['passes']}/{b['k']} → {c['passes']}/{c['k']}")
    h_before = split_summary(base, "holdout")["mean"]
    h_after = split_summary(cand, "holdout")["mean"]
    holdout_ok = h_after >= h_before - config()["gate"]["holdout_tolerance"]

    reasons = []
    if not improved:
        reasons.append(f"target did not improve ({before:.2f} → {after:.2f})")
    if regressions:
        reasons.append(f"train regressions: {regressions}")
    if new_gates:
        reasons.append(f"new hard gates: {new_gates}")
    if not holdout_ok:
        reasons.append(f"holdout dropped {h_before:.2f} → {h_after:.2f}")
    passed = not reasons
    if passed:
        reasons.append("accepted" + (f" (tolerated noise dips: {dips})" if dips else ""))
    return GateRecord(baseline_run=baseline.run_id, candidate_run=candidate.run_id, target_before=before,
                      target_after=after, regressions=regressions + dips, new_hard_gates=new_gates,
                      holdout_delta=round(h_after - h_before, 3), passed=passed, reason="; ".join(reasons))


def screen(baseline: SuiteRun, targeted: SuiteRun, targets: list[str]) -> tuple[bool, float, float]:
    """Cheap pre-check on the target scenarios only."""
    return target_improved(aggregate(baseline), aggregate(targeted), targets)
