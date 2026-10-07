"""`uv run chat` — talk to the scheduling agent as a patient."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.panel import Panel

from agent.ehr import MockEHR
from agent.loop import Agent
from llm import LLM, BudgetExceeded, ledger_total
from memory.store import LessonStore

console = Console()
app = typer.Typer(add_completion=False)


@app.command()
def chat(show_trace: bool = typer.Option(False, help="Print session state after each turn."),
         no_lessons: bool = typer.Option(False, help="Run the v0 agent without learned lessons."),
         budget_usd: float = typer.Option(0.5, help="Spend cap for this chat session.")):
    lessons = [] if no_lessons else LessonStore().accepted()
    ehr = MockEHR()
    llm = LLM(run_id="chat", run_cap=budget_usd, use_cache=False)

    def on_tool(name, args, result):
        ok = "[green]ok[/]" if result.get("ok") else f"[red]{result.get('error')}[/]"
        console.print(f"  [dim]⚙ {name}({json.dumps(args)[:140]}) → [/]{ok}")

    agent = Agent(ehr, llm, lessons, on_tool=on_tool)
    console.print(Panel.fit(f"[bold]{ehr.clinic['name']}[/] — scheduling assistant\n"
                            f"[dim]{len(lessons)} learned lesson(s) active · type /quit to exit[/]"))
    console.print("[bold cyan]Agent:[/] Thanks for calling Riverside Family Health. How can I help you today?")
    while True:
        try:
            user = console.input("[bold green]You:[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user in {"/quit", "/exit"}:
            break
        try:
            reply = agent.step(user)
        except BudgetExceeded as e:
            console.print(f"[red]Budget stop: {e}[/]")
            break
        console.print(f"[bold cyan]Agent:[/] {reply}")
        if show_trace:
            console.print(f"[dim]{agent.session.to_dict()}[/]")
    console.print(f"[dim]session cost ${llm.spent:.4f} · ledger total ${ledger_total():.3f}[/]")


def main():
    app()
