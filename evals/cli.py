"""`uv run evals ...` — run the suite, inspect reports, transcripts and spend."""

from __future__ import annotations

import json
from collections import defaultdict

import typer
from rich.console import Console

from evals.report import print_table, write_report
from evals.runner import RUNS_DIR, load_run, run_suite
from evals.scenario import load_scenarios
from llm import LEDGER, LLM, BudgetExceeded, ledger_total
from memory.store import LessonStore

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


@app.command()
def run(split: str = typer.Option("all", help="train | holdout | all"),
        k: int = typer.Option(1, help="Runs per scenario."),
        only: str = typer.Option("", help="Comma-separated scenario id prefixes, e.g. T05,T07"),
        no_lessons: bool = typer.Option(False, help="Ignore learned lessons (v0 baseline)."),
        no_cache: bool = typer.Option(False, help="Disable the dev response cache."),
        budget_usd: float = typer.Option(1.0, help="Spend cap for this run."),
        label: str = typer.Option("", help="Suffix for the run id.")):
    """Run scenarios and print a scored table."""
    scenarios = load_scenarios(split, [o for o in only.split(",") if o] or None)
    lessons = [] if no_lessons else LessonStore().accepted()
    llm = LLM(run_id=label or "evals", run_cap=budget_usd, use_cache=not no_cache)
    console.print(f"Running {len(scenarios)} scenario(s) × k={k} with {len(lessons)} lesson(s)…")
    try:
        result = run_suite(scenarios, lessons, llm, k, split, label)
    except BudgetExceeded as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    print_table(result, console=console)
    console.print(f"report: {write_report(result)} · run cost ${llm.spent:.4f} · ledger ${ledger_total():.3f}")


@app.command()
def report(run_id: str, compare: str = typer.Option("", help="Baseline run id for before/after.")):
    """Re-render a saved run (optionally against a baseline)."""
    r = load_run(run_id)
    before = load_run(compare) if compare else None
    print_table(r, before, console)
    console.print(write_report(r, before))


@app.command()
def show(run_id: str, scenario: str, idx: int = 0):
    """Print one transcript with its tool calls and check results."""
    d = json.loads((RUNS_DIR / run_id / "transcripts" / f"{scenario}#{idx}.json").read_text())
    tools = defaultdict(list)
    for t in d["trace"]:
        tools[t["turn"]].append(t)
    for m in d["transcript"]:
        if m["role"] == "patient":
            heard = f" [dim](heard: {m['text']})[/]" if m["text"] != m.get("said", m["text"]) else ""
            console.print(f"[green]Patient:[/] {m.get('said', m['text'])}{heard}")
        else:
            for t in tools.pop(m["turn"], []):
                if t.get("tool"):
                    ok = "ok" if t["result"].get("ok") else t["result"].get("error")
                    console.print(f"  [dim]⚙ {t['tool']}({json.dumps(t['args'])[:120]}) → {ok}[/]")
                else:
                    console.print(f"  [magenta]⚡ {t.get('event')}[/]")
            console.print(f"[cyan]Agent:[/] {m['text']}")
    s = d["score"]
    console.print(f"\n[bold]total {s.get('total')}[/] state {s.get('state')} trace {s.get('trace')} "
                  f"judge {s.get('judge_score')} · ended_by {d['ended_by']} {d.get('error') or ''}")
    for g, ev in s.get("hard_gates", {}).items():
        console.print(f"[red]HARD GATE {g}: {ev}[/]")
    for c in s.get("checks", []):
        mark = "[green]✓[/]" if c["passed"] else "[red]✗[/]"
        console.print(f" {mark} {c['id']}: {c['detail'][:150]}")


@app.command()
def cost():
    """Total OpenAI spend recorded in the ledger, by role."""
    by = defaultdict(float)
    if LEDGER.exists():
        for line in LEDGER.read_text().splitlines():
            e = json.loads(line)
            by[f"{e['role']} ({e['model']})"] += e["usd"]
    for k, v in sorted(by.items()):
        console.print(f"  {k:<28} ${v:.4f}")
    console.print(f"[bold]total ${ledger_total():.4f}[/] of $7.00 cap")
