"""The self-improvement loop: baseline → diagnose → screen → gate → accept/reject → report."""

from __future__ import annotations

import difflib

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from agent.ehr import MockEHR
from agent.prompt import build_system_prompt
from agent.tools import BASE_DESCRIPTIONS
from evals.report import aggregate, print_table, render_markdown, split_summary, write_report
from evals.runner import RUNS_DIR, SuiteRun, latest_matching_run, load_run, new_run_id, run_suite
from evals.scenario import load_scenarios
from improve.diagnose import cluster_failures, propose
from improve.gate import evaluate, screen
from llm import LLM, BudgetExceeded, ledger_total
from memory.store import Lesson, LessonStore

console = Console()


def _playbook_diff(before: list[Lesson], after: list[Lesson]) -> str:
    ehr = MockEHR()
    a = build_system_prompt(ehr, before).splitlines()
    b = build_system_prompt(ehr, after).splitlines()
    return "\n".join(l for l in difflib.unified_diff(a, b, lineterm="", n=0) if l[:1] in "+-" and l[:3] not in {"+++", "---"})


def _effective(accepted: list[Lesson], new: list[Lesson]) -> list[Lesson]:
    replaced = {m for l in new for m in l.merged_from}
    return [l for l in accepted if l.id not in replaced] + new


def improve(rounds: int = 1, k: int = 3, budget_usd: float = 2.0, approve: bool = False,
            use_cache: bool = True, baseline_id: str | None = None, fresh_baseline: bool = False,
            max_candidates: int = 3) -> str:
    loop_id = new_run_id("loop")
    store = LessonStore()
    llm = LLM(run_id=loop_id, run_cap=budget_usd, use_cache=use_cache)
    scenarios = load_scenarios("all")
    train = [s for s in scenarios if s.split == "train"]
    by_id = {s.id: s for s in scenarios}
    log: list[str] = []
    first_baseline: SuiteRun | None = None
    baseline: SuiteRun | None = None

    try:
        if baseline_id:
            baseline = load_run(baseline_id)
            console.print(f"[bold]Baseline[/] using run {baseline_id} (given explicitly)")
        elif not fresh_baseline:
            baseline = latest_matching_run({s.id for s in scenarios}, store.accepted(), k)
            if baseline:
                console.print(f"[bold]Baseline[/] reusing latest matching run [cyan]{baseline.run_id}[/] "
                              f"(same prompt + lessons, k={k}, all {len(scenarios)} scenarios). "
                              f"Use --fresh-baseline to re-run it.")
            else:
                console.print(f"[yellow]No saved full run matches the current agent at k={k}; "
                              f"running a fresh baseline.[/]")
        if baseline is None:
            console.rule("[bold]Baseline run")
            baseline = run_suite(scenarios, store.accepted(), llm, k, "all", f"{loop_id}-baseline")
            write_report(baseline)
        first_baseline = baseline
        print_table(baseline, console=console)

        for rnd in range(1, rounds + 1):
            console.rule(f"[bold]Round {rnd}: diagnose failures")
            clusters = cluster_failures(baseline, max_candidates)
            if not clusters:
                console.print("[green]No train failures left to learn from.[/]")
                break
            accepted = store.accepted()
            prompt = build_system_prompt(MockEHR(), accepted)
            candidates: list[tuple[Lesson, list[str]]] = []
            for cl in clusters:
                console.print(f"• cluster [bold]{cl['key']}[/] ({len(cl['results'])} failed runs: "
                              f"{', '.join(cl['scenarios'])})")
                lesson = propose(cl, prompt, BASE_DESCRIPTIONS, store.lessons, train, llm, store.next_id())
                if not lesson:
                    console.print("  [yellow]optimizer could not produce a general (lint-clean) lesson[/]")
                    continue
                lesson.source.run_id = baseline.run_id
                store.add(lesson)
                console.print(Panel(f"[bold]{lesson.id}[/] → {lesson.target}"
                                    f"{' (merges ' + ', '.join(lesson.merged_from) + ')' if lesson.merged_from else ''}\n"
                                    f"[cyan]When[/] {lesson.trigger}: {lesson.rule}\n[dim]root cause: {lesson.root_cause}[/]",
                                    title="proposed lesson", expand=False))
                if lesson.target == "needs_code":
                    lesson.status = "needs_human"
                    log.append(f"{lesson.id}: needs_code → flagged for a human (not auto-applied)")
                    console.print("  [magenta]needs a code guardrail → flagged for human review, not applied[/]")
                    continue
                candidates.append((lesson, cl["scenarios"]))
            store.save()

            # Screen: each candidate on its own cluster's scenarios only (cheap).
            screened: list[tuple[Lesson, list[str]]] = []
            for lesson, targets in candidates:
                console.rule(f"Screen {lesson.id} on {', '.join(targets)}")
                trial = run_suite([by_id[t] for t in targets], _effective(accepted, [lesson]), llm, k,
                                  "train", f"{loop_id}-screen-{lesson.id}")
                ok, before, after = screen(baseline, trial, targets)
                console.print(f"  target pass rate {before:.2f} → {after:.2f}  "
                              f"{'[green]promising[/]' if ok else '[red]no improvement → rejected[/]'}")
                if ok:
                    screened.append((lesson, targets))
                else:
                    lesson.status = "rejected"
                    log.append(f"{lesson.id}: rejected at screening ({before:.2f} → {after:.2f})")
            store.save()
            if not screened:
                console.print("[yellow]No candidate survived screening this round.[/]")
                continue

            # Gate: all screened candidates together on the FULL train + holdout suite;
            # if the combination fails, bisect by gating each candidate on its own.
            def gate(group, final: bool):
                nonlocal baseline
                ids = ", ".join(l.id for l, _ in group)
                console.rule(f"Gate {ids} on full suite (train + holdout)")
                trial_lessons = _effective(store.accepted(), [l for l, _ in group])
                gate_run = run_suite(scenarios, trial_lessons, llm, k, "all", f"{loop_id}-gate-r{rnd}")
                record = evaluate(baseline, gate_run, sorted({t for _, ts in group for t in ts}))
                print_table(gate_run, baseline, console)
                console.print(f"  gate: {'[green]PASS[/]' if record.passed else '[red]FAIL[/]'} — {record.reason}")
                if record.passed and approve:
                    console.print(Panel(_playbook_diff(store.accepted(), trial_lessons) or "(tool notes only)",
                                        title="prompt diff"))
                    if not Confirm.ask("Accept these lessons?", default=True):
                        record.passed, record.reason = False, "rejected by human reviewer"
                write_report(gate_run, baseline)
                for l, _ in group:
                    l.gate = record
                    if record.passed:
                        l.status = "accepted"
                        for old in l.merged_from:
                            store.set_status(old, "retired")
                        log.append(f"{l.id}: ACCEPTED — {record.reason}")
                    elif final:
                        l.status = "rejected"
                        log.append(f"{l.id}: rejected at gate — {record.reason}")
                store.save()
                if record.passed:
                    baseline = gate_run
                return record.passed

            if not gate(screened, final=len(screened) == 1) and len(screened) > 1:
                for cand in screened:
                    gate([cand], final=True)
    except BudgetExceeded as e:
        console.print(f"[red]Budget stop: {e}[/] — saving partial results")
        log.append(f"stopped early: {e}")
        store.save()

    return _final_report(loop_id, first_baseline, baseline, store, log, llm)


def _final_report(loop_id, first: SuiteRun | None, last: SuiteRun | None, store: LessonStore,
                  log: list[str], llm) -> str:
    out_dir = RUNS_DIR / loop_id
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# Improvement loop `{loop_id}`", "", f"Spend this loop: ${llm.spent:.3f} · ledger total "
             f"${ledger_total():.3f} of $7.00", "", "## Decisions", *[f"- {x}" for x in log], ""]
    if first and last:
        a, b = aggregate(first), aggregate(last)
        for split in ("train", "holdout"):
            s0, s1 = split_summary(a, split), split_summary(b, split)
            lines.append(f"- **{split}**: mean {s0['mean']:.2f} → {s1['mean']:.2f}, "
                         f"pass rate {s0['pass_rate']:.0%} → {s1['pass_rate']:.0%}")
        lines += ["", render_markdown(last, first) if last is not first else render_markdown(first)]
        console.rule("[bold]Before → after")
        if last is not first:
            print_table(last, first, console)
        for split in ("train", "holdout"):
            s0, s1 = split_summary(a, split), split_summary(b, split)
            console.print(f"  [bold]{split}[/]: mean {s0['mean']:.2f} → {s1['mean']:.2f} · "
                          f"pass rate {s0['pass_rate']:.0%} → {s1['pass_rate']:.0%}")
    lines += ["", "## Lessons", ""]
    for l in store.lessons:
        lines.append(f"- **{l.id}** [{l.status}] ({l.target}) When {l.trigger}: {l.rule}"
                     + (f"  \n  gate: {l.gate.reason}" if l.gate else ""))
    path = out_dir / "loop_report.md"
    path.write_text("\n".join(lines) + "\n")
    for x in log:
        console.print(f"  • {x}")
    console.print(f"[dim]loop report: {path} · spent ${llm.spent:.3f} · ledger ${ledger_total():.3f}[/]")
    return str(path)
