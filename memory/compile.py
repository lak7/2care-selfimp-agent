"""Compile accepted lessons into the prompt Playbook and tool-description notes."""

from __future__ import annotations


def _active(lessons: list) -> list:
    """Lessons that should be compiled. A merged lesson supersedes the ones it replaced."""
    superseded = {m for l in lessons for m in l.merged_from}
    return sorted((l for l in lessons if l.id not in superseded), key=lambda l: l.id)


def playbook_text(lessons: list) -> str:
    rules = [l for l in _active(lessons) if l.target == "playbook"]
    if not rules:
        return ""
    lines = ["## 5. Playbook (learned from past calls)"]
    lines += [f"- [{l.id}] When {l.trigger.rstrip('.')}: {l.rule}" for l in rules]
    return "\n".join(lines) + "\n"


def tool_notes(lessons: list) -> dict[str, list[str]]:
    notes: dict[str, list[str]] = {}
    for l in _active(lessons):
        if l.target.startswith("tool_desc:"):
            notes.setdefault(l.target.split(":", 1)[1], []).append(l.rule)
    return notes
