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
from improve.gate import evaluate, merge_runs, needs_confirmation, screen
from llm import LLM, BudgetExceeded, ledger_total
from memory.compile import clean_trigger
from memory.store import Lesson, LessonSource, LessonStore

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
            max_candidates: int = 3, profile: str = "full") -> str:
    loop_id = new_run_id("loop")
    store = LessonStore()
    llm = LLM(run_id=loop_id, run_cap=budget_usd, use_cache=use_cache)
    scenarios = load_scenarios("all", profile=profile)
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
            # Skip failure clusters whose lesson was already rejected or sent to a human, so a
            # second round tries the next most common failure instead of the same one.
            tried = {l.cluster for l in store.lessons if l.cluster and l.status in ("rejected", "needs_human")}
            clusters = [c for c in cluster_failures(baseline, max_candidates + len(tried))
                        if c["key"] not in tried][:max_candidates]
            if tried:
                console.print(f"  [dim]skipping clusters already tried: {', '.join(sorted(tried))}[/]")
            if not clusters:
                console.print("[green]No train failures left to learn from.[/]")
                break
            _route_style_failures(baseline, store, log)
            accepted = store.accepted()
            prompt = build_system_prompt(MockEHR(), accepted)
            candidates: list[tuple[Lesson, list[str]]] = []
            for cl in clusters:
                console.print(f"• cluster [bold]{cl['key']}[/] ({len(cl['results'])} failed runs: "
                              f"{', '.join(cl['scenarios'])})")
                lesson, problems = propose(cl, prompt, BASE_DESCRIPTIONS, store.lessons, train, llm,
                                           store.next_id(), baseline)
                if not lesson:
                    console.print(f"  [yellow]optimizer could not produce a general (lint-clean) lesson: "
                                  f"{'; '.join(problems)}[/]")
                    log.append(f"cluster {cl['key']}: no lesson (lint: {'; '.join(problems)})")
                    continue
                lesson.source.run_id = baseline.run_id
                store.add(lesson)
                console.print(Panel(f"[bold]{lesson.id}[/] → {lesson.target}"
                                    f"{' (merges ' + ', '.join(lesson.merged_from) + ')' if lesson.merged_from else ''}\n"
                                    f"[cyan]When[/] {clean_trigger(lesson.trigger)}: {lesson.rule}\n[dim]root cause: {lesson.root_cause}[/]",
                                    title="proposed lesson", expand=False))
                if lesson.target == "needs_code":
                    lesson.status = "needs_human"
                    log.append(f"{lesson.id}: needs_code → flagged for a human (not auto-applied)")
                    console.print("  [magenta]needs a code guardrail → flagged for human review, not applied[/]")
                    continue
                candidates.append((lesson, cl["scenarios"]))
            store.save()

            # Screen: each candidate on its own cluster's scenarios only (cheap). With a single
            # candidate the gate run answers the same question, so screening is skipped.
            screened: list[tuple[Lesson, list[str]]] = []
            if len(candidates) == 1:
                screened, candidates = candidates, []
                console.print("  single candidate → skipping screening, going straight to the gate")
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
                targets = sorted({t for _, ts in group for t in ts})
                record = evaluate(baseline, gate_run, targets)
                print_table(gate_run, baseline, console)
                console.print(f"  gate: {'[green]PASS[/]' if record.passed else '[red]FAIL[/]'} — {record.reason}")
                recheck = needs_confirmation(record, targets, baseline, gate_run)
                if recheck:
                    # Borderline verdict: re-run just the scenarios it hinged on, with and without the
                    # lesson, and decide on the pooled samples instead of one noisy run each.
                    console.rule(f"Confirm: {k} more runs of {', '.join(recheck)} (with and without {ids})")
                    extra = [by_id[s] for s in recheck]
                    more_b = run_suite(extra, store.accepted(), llm, k, "all", f"{loop_id}-confirm-base-r{rnd}",
                                       idx_offset=100)
                    more_c = run_suite(extra, trial_lessons, llm, k, "all", f"{loop_id}-confirm-cand-r{rnd}",
                                       idx_offset=100)
                    baseline_pooled, gate_pooled = merge_runs(baseline, more_b), merge_runs(gate_run, more_c)
                    record = evaluate(baseline_pooled, gate_pooled, targets)
                    record.confirmed = True
                    print_table(gate_pooled, baseline_pooled, console)
                    console.print(f"  confirmed gate: {'[green]PASS[/]' if record.passed else '[red]FAIL[/]'}"
                                  f" — {record.reason}")
                    if record.passed:
                        gate_run = gate_pooled
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


def _route_style_failures(baseline: SuiteRun, store: LessonStore, log: list[str]) -> None:
    """Formatting/tone failures go to a human as a code fix (text-to-speech normalisation), once."""
    from evals.checks import STYLE_CHECKS
    from improve.diagnose import failure_ids

    hits = [r for r in baseline.results if r.split == "train" and STYLE_CHECKS & set(failure_ids(r))]
    if not hits or any(l.cluster == "style" for l in store.lessons):
        return
    lesson = Lesson(id=store.next_id(), source=LessonSource(scenario=hits[0].scenario_id, run_id=baseline.run_id),
                    violated=sorted(STYLE_CHECKS & {i for r in hits for i in failure_ids(r)}),
                    evidence=f"{len(hits)} failed train runs have formatting/tone failures (markdown, long replies, "
                             f"several questions per turn).",
                    root_cause="The base prompt already asks for short spoken sentences; the model does not "
                               "follow it reliably, and prompt rules about style apply to every turn.",
                    target="needs_code", trigger="the agent's reply is about to be spoken",
                    rule="Add a text-to-speech formatting step in code: strip markdown and lists, split or "
                         "shorten replies over ~60 words, keep one question per turn.",
                    cluster="style", status="needs_human")
    store.add(lesson)
    store.save()
    log.append(f"{lesson.id}: style failures in {len(hits)} runs → needs_code (TTS formatting layer), not a prompt rule")
    console.print(f"  [magenta]{lesson.id}: style failures ({len(hits)} runs) → flagged for a code fix "
                  f"(TTS formatting), not learned as a prompt rule[/]")


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
        lines.append(f"- **{l.id}** [{l.status}] ({l.target}) When {clean_trigger(l.trigger)}: {l.rule}"
                     + (f"  \n  gate: {l.gate.reason}" if l.gate else ""))
    path = out_dir / "loop_report.md"
    path.write_text("\n".join(lines) + "\n")
    for x in log:
        console.print(f"  • {x}")
    console.print(f"[dim]loop report: {path} · spent ${llm.spent:.3f} · ledger ${ledger_total():.3f}[/]")
    return str(path)
