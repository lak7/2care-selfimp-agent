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
    assert not rec.passed and "HG1_phi_leak" in rec.reason   # total gate hits 0 -> 1


def test_rejects_holdout_drop():
    cand = {**BASE, "T1": ("train", [1, 1, 1], []), "H1": ("holdout", [0.7, 0.7, 0.7], [])}
    assert not evaluate(suite(BASE), suite(cand, "c"), ["T1"]).passed


def test_one_off_gate_offset_by_a_fixed_one_is_tolerated():
    base = {**BASE, "H1": ("holdout", [0, 0.9, 0.9], ["HG4_missed_emergency"])}
    cand = {**base, "T1": ("train", [1, 1, 1], []), "H1": ("holdout", [0.9, 0.9, 0.9], []),
            "T2": ("train", [0, 1, 1], ["HG5_hallucination"])}
    rec = evaluate(suite(base), suite(cand, "c"), ["T1"])
    assert rec.passed and "hard-gate conversations 1 → 1" in rec.reason


def test_systematic_new_gate_rejected():
    cand = {**BASE, "T1": ("train", [1, 1, 1], []), "T2": ("train", [0, 0, 1], [])}
    s = suite(cand, "c")
    for r in s.results:
        if r.scenario_id == "T2" and r.run_idx < 2:
            r.score["hard_gates"] = {"HG5_hallucination": ["x"]}
    base = suite({**BASE, "H1": ("holdout", [0.9, 0.9, 0.9], ["HG1_phi_leak"])})
    for r in base.results:
        if r.scenario_id == "H1":
            r.score["hard_gates"] = {"HG1_phi_leak": ["x"]}   # baseline: 3 hits, so the total doesn't rise
    rec = evaluate(base, s, ["T1"])
    assert not rec.passed and "systematic" in rec.reason


def test_style_checks_never_become_prompt_clusters():
    from improve.diagnose import cluster_failures
    run = suite({"T1": ("train", [0.5, 0.5, 0.5], [])})
    for r in run.results:
        r.score["checks"] = [{"id": "trace:voice_ready", "layer": "trace", "passed": False, "detail": "md"},
                             {"id": "trace:fee_disclosed_before_cancel", "layer": "trace", "passed": False, "detail": "x"}]
    assert [c["key"] for c in cluster_failures(run)] == ["trace:fee_disclosed_before_cancel"]


def test_borderline_verdicts_get_confirmed():
    from improve.gate import needs_confirmation
    cand = {**BASE, "T1": ("train", [1, 1, 1], []), "T2": ("train", [1, 0, 0], [])}
    b, c = suite(BASE), suite(cand, "c")
    rec = evaluate(b, c, ["T1"])                         # target up, but T2 regressed
    assert not rec.passed and needs_confirmation(rec, ["T1"], b, c) == ["T1", "T2"]
    flat = evaluate(b, suite(BASE, "c"), ["T1"])         # target didn't move: clear-cut, no re-run
    assert needs_confirmation(flat, ["T1"], b, suite(BASE, "c")) == []
