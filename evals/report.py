"""Aggregate suite runs into per-scenario stats, before/after comparisons and markdown reports."""

from __future__ import annotations

from collections import Counter, defaultdict

from rich.console import Console
from rich.table import Table

from evals.runner import RUNS_DIR, SuiteRun


def aggregate(run: SuiteRun) -> dict[str, dict]:
    by: dict[str, list] = defaultdict(list)
    for r in run.results:
        by[r.scenario_id].append(r)
    out = {}
    for sid, rs in sorted(by.items()):
        totals = [r.score.get("total", 0.0) for r in rs]
        failed = Counter(c["id"] for r in rs for c in r.score.get("checks", []) if not c["passed"])
        gates = Counter(g for r in rs for g in r.score.get("hard_gates", {}))
        out[sid] = {
            "split": rs[0].split, "k": len(rs),
            "passes": sum(r.score.get("passed", False) for r in rs),
            "pass_rate": sum(r.score.get("passed", False) for r in rs) / len(rs),
            "mean": sum(totals) / len(totals), "min": min(totals), "max": max(totals),
            "hard_gates": dict(gates), "failed_checks": dict(failed),
            "errors": [r.error for r in rs if r.error],
        }
    return out


def split_summary(agg: dict, split: str) -> dict:
    rows = [v for v in agg.values() if v["split"] == split]
    if not rows:
        return {"mean": 0.0, "pass_rate": 0.0, "n": 0}
    return {"mean": sum(v["mean"] for v in rows) / len(rows),
            "pass_rate": sum(v["pass_rate"] for v in rows) / len(rows), "n": len(rows)}


def _arrow(a: float, b: float) -> str:
    return "↑" if b > a + 1e-9 else "↓" if b < a - 1e-9 else "="


def render_markdown(run: SuiteRun, before: SuiteRun | None = None) -> str:
    agg = aggregate(run)
    pre = aggregate(before) if before else {}
    lines = [f"# Eval run `{run.run_id}`", "",
             f"- split: {run.split} · k={run.k} · prompt `{run.prompt_hash}` · lessons: "
             f"{', '.join(run.lesson_ids) or 'none'} · cost ${run.cost:.3f}", ""]
    for split in ("train", "holdout"):
        s = split_summary(agg, split)
        if s["n"]:
            extra = ""
            if before:
                b = split_summary(pre, split)
                extra = f" (before: mean {b['mean']:.2f}, pass {b['pass_rate']:.0%})"
            lines.append(f"- **{split}**: mean {s['mean']:.2f}, pass rate {s['pass_rate']:.0%}{extra}")
    lines += ["", "| scenario | split | pass | mean | hard gates | failing checks |", "|---|---|---|---|---|---|"]
    for sid, v in agg.items():
        p = f"{v['passes']}/{v['k']}"
        m = f"{v['mean']:.2f}"
        if sid in pre:
            b = pre[sid]
            p = f"{b['passes']}/{b['k']} → {p} {_arrow(b['pass_rate'], v['pass_rate'])}"
            m = f"{b['mean']:.2f} → {m}"
        gates = ", ".join(f"{g}×{n}" for g, n in v["hard_gates"].items()) or "—"
        fails = ", ".join(f"{c}×{n}" for c, n in sorted(v["failed_checks"].items())) or "—"
        lines.append(f"| {sid} | {v['split']} | {p} | {m} | {gates} | {fails} |")
    return "\n".join(lines) + "\n"


def print_table(run: SuiteRun, before: SuiteRun | None = None, console: Console | None = None) -> None:
    console = console or Console()
    agg, pre = aggregate(run), (aggregate(before) if before else {})
    t = Table(title=f"run {run.run_id}  ·  k={run.k}  ·  ${run.cost:.3f}")
    t.add_column("scenario")
    t.add_column("split")
    if before:
        t.add_column("before", justify="right")
    t.add_column("pass", justify="right")
    t.add_column("mean", justify="right")
    t.add_column("hard gates / failing checks", overflow="fold")
    for sid, v in agg.items():
        color = "green" if v["pass_rate"] == 1 else "yellow" if v["pass_rate"] > 0 else "red"
        row = [sid, v["split"]]
        if before:
            b = pre.get(sid)
            row.append(f"{b['passes']}/{b['k']}" if b else "-")
            if b and v["pass_rate"] < b["pass_rate"]:
                color = "bold red"
        problems = list(v["hard_gates"]) + [c for c in v["failed_checks"] if not c.startswith("judge:")]
        row += [f"[{color}]{v['passes']}/{v['k']}[/]", f"{v['mean']:.2f}", ", ".join(problems)[:90] or "—"]
        t.add_row(*row)
    console.print(t)
    for split in ("train", "holdout"):
        s = split_summary(agg, split)
        if s["n"]:
            console.print(f"  {split}: mean {s['mean']:.2f}, pass rate {s['pass_rate']:.0%}")


def write_report(run: SuiteRun, before: SuiteRun | None = None) -> str:
    path = RUNS_DIR / run.run_id / "report.md"
    path.write_text(render_markdown(run, before))
    return str(path)
