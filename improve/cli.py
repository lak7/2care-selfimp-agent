"""`uv run improve` — run the full self-improvement loop."""

from __future__ import annotations

import typer

from improve.loop import improve

app = typer.Typer(add_completion=False)


@app.command()
def run(rounds: int = typer.Option(1, help="Diagnose → gate rounds."),
        k: int = typer.Option(3, help="Runs per scenario (noise control)."),
        budget_usd: float = typer.Option(2.0, help="Hard spend cap for this loop."),
        approve: bool = typer.Option(False, help="Pause for human approval before accepting lessons."),
        no_cache: bool = typer.Option(False, help="Disable the dev response cache (use for recorded runs)."),
        baseline: str = typer.Option("", help="Reuse an existing run id as the baseline."),
        max_candidates: int = typer.Option(3, help="Max failure clusters / lessons per round.")):
    """Baseline → diagnose failures → propose lessons → screen → gate → accept/reject → report."""
    improve(rounds=rounds, k=k, budget_usd=budget_usd, approve=approve, use_cache=not no_cache,
            baseline_id=baseline or None, max_candidates=max_candidates)


def main():
    app()
