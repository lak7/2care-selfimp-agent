"""`uv run memory ...` — inspect, retire and ablate learned lessons."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from evals.report import print_table, split_summary, aggregate
from evals.runner import run_suite
from evals.scenario import load_scenarios
from llm import LLM
from memory.store import LessonStore

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


@app.command("list")
def list_():
    """All lessons with status and gate outcome."""
    t = Table()
    for col in ("id", "status", "target", "rule", "gate"):
        t.add_column(col, overflow="fold")
    for l in LessonStore().lessons:
        t.add_row(l.id, l.status, l.target, f"When {l.trigger}: {l.rule}", l.gate.reason if l.gate else "—")
    console.print(t)


@app.command()
def show(lesson_id: str):
    console.print_json(LessonStore().get(lesson_id).model_dump_json())


@app.command()
def retire(lesson_id: str):
    """Remove a lesson from the compiled prompt (reversible: set it back with `restore`)."""
    store = LessonStore()
    store.set_status(lesson_id, "retired")
    store.save()
    console.print(f"{lesson_id} retired")


@app.command()
def restore(lesson_id: str):
    store = LessonStore()
    store.set_status(lesson_id, "accepted")
    store.save()
    console.print(f"{lesson_id} accepted again")


@app.command()
def reset(yes: bool = typer.Option(False, "--yes", help="Confirm wiping all lessons.")):
    """Clear memory back to the v0 agent (used before recording a demo)."""
    if not yes:
        console.print("pass --yes to wipe memory/lessons.json")
        raise typer.Exit(1)
    store = LessonStore()
    store.lessons = []
    store.save()
    console.print("memory cleared")


@app.command()
def ablate(lesson_id: str, k: int = 3, split: str = "all", budget_usd: float = 1.0):
    """Measure a lesson's contribution: run the suite with and without it."""
    store = LessonStore()
    with_l = store.accepted()
    without = [l for l in with_l if l.id != lesson_id]
    llm = LLM(run_id=f"ablate-{lesson_id}", run_cap=budget_usd)
    scs = load_scenarios(split)
    a = run_suite(scs, with_l, llm, k, split, f"ablate-{lesson_id}-with")
    b = run_suite(scs, without, llm, k, split, f"ablate-{lesson_id}-without")
    print_table(a, b, console)
    for s in ("train", "holdout"):
        x, y = split_summary(aggregate(b), s), split_summary(aggregate(a), s)
        console.print(f"{s}: without {x['mean']:.2f} → with {y['mean']:.2f}")
