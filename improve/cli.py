"""`uv run improve` — run the full self-improvement loop."""

from __future__ import annotations

import typer

from improve.loop import improve
from llm import config

app = typer.Typer(add_completion=False)


@app.command()
def run(rounds: int = typer.Option(0, help="Diagnose → gate rounds (default: the profile's, else 1)."),
        profile: str = typer.Option("full", help="Scenario set from config.yaml: full | demo"),
        k: int = typer.Option(0, help="Runs per scenario (default: the profile's k)."),
        budget_usd: float = typer.Option(2.0, help="Hard spend cap for this loop."),
        approve: bool = typer.Option(False, "--approve", help="Pause for human approval before accepting lessons."),
        no_cache: bool = typer.Option(False, "--no-cache", help="Disable the dev response cache (use for recorded runs)."),
        baseline: str = typer.Option("", help="Use this specific run id as the baseline."),
        fresh_baseline: bool = typer.Option(False, "--fresh-baseline", help="Ignore saved runs and re-run the baseline from scratch."),
        max_candidates: int = typer.Option(0, help="Max failure clusters / lessons per round (default: profile's).")):
    """Diagnose the latest matching eval run → propose lessons → screen → gate → accept/reject → report.

    Baseline: by default the most recent saved `evals run` whose prompt + lessons and k match the
    current agent (so run `uv run evals run` with the same --profile first). If none matches, or with
    --fresh-baseline, the baseline is run as part of the loop."""
    improve(rounds=rounds or config()["profiles"][profile].get("rounds", 1), k=k or config()["profiles"][profile]["k"], profile=profile, budget_usd=budget_usd, approve=approve, use_cache=not no_cache,
            baseline_id=baseline or None, fresh_baseline=fresh_baseline,
            max_candidates=max_candidates or config()["profiles"][profile].get("max_candidates", 3))


def main():
    app()
