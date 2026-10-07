"""Gate decisions on synthetic suite results (no LLM)."""

from evals.runner import Result, SuiteRun
from improve.gate import evaluate


def suite(spec: dict[str, tuple[str, list[float], list[str]]], rid="r") -> SuiteRun:
    """spec: scenario -> (split, per-run totals, hard gates present in run 0)."""
    results = []
    for sid, (split, totals, gates) in spec.items():
        for i, t in enumerate(totals):
            hg = {g: ["x"] for g in gates} if i == 0 else {}
            results.append(Result(scenario_id=sid, split=split, run_idx=i,
                                  score={"total": t, "passed": t >= 0.8, "hard_gates": hg, "checks": []}))
    return SuiteRun(run_id=rid, created_at="", split="all", k=3, prompt_hash="", lesson_ids=[], results=results)


BASE = {"T1": ("train", [0, 0, 0.9], []), "T2": ("train", [1, 1, 1], []), "H1": ("holdout", [0.9, 0.9, 0.9], [])}


def test_accepts_clean_improvement():
    cand = {**BASE, "T1": ("train", [0.9, 0.9, 0.9], [])}
    assert evaluate(suite(BASE), suite(cand, "c"), ["T1"]).passed


def test_rejects_when_target_does_not_improve():
    assert not evaluate(suite(BASE), suite(BASE, "c"), ["T1"]).passed


def test_rejects_train_regression():
    cand = {**BASE, "T1": ("train", [1, 1, 1], []), "T2": ("train", [1, 0, 0], [])}
    rec = evaluate(suite(BASE), suite(cand, "c"), ["T1"])
    assert not rec.passed and "regressions" in rec.reason


def test_tolerates_single_run_dip_but_logs_it():
    cand = {**BASE, "T1": ("train", [1, 1, 1], []), "T2": ("train", [1, 1, 0], [])}
    rec = evaluate(suite(BASE), suite(cand, "c"), ["T1"])
    assert rec.passed and rec.regressions == ["T2: 3/3 → 2/3"]


def test_rejects_new_hard_gate_anywhere():
    cand = {**BASE, "T1": ("train", [1, 1, 1], []), "H1": ("holdout", [0, 0.9, 0.9], ["HG1_phi_leak"])}
    rec = evaluate(suite(BASE), suite(cand, "c"), ["T1"])
    assert not rec.passed and "HG1_phi_leak" in rec.reason


def test_rejects_holdout_drop():
    cand = {**BASE, "T1": ("train", [1, 1, 1], []), "H1": ("holdout", [0.7, 0.7, 0.7], [])}
    assert not evaluate(suite(BASE), suite(cand, "c"), ["T1"]).passed
