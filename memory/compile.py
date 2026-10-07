"""Compile accepted lessons into the prompt Playbook and tool-description notes."""

from __future__ import annotations


def _active(lessons: list) -> list:
    """Lessons that should be compiled. A merged lesson supersedes the ones it replaced."""
    superseded = {m for l in lessons for m in l.merged_from}
    return sorted((l for l in lessons if l.id not in superseded), key=lambda l: l.id)


def clean_trigger(trigger: str) -> str:
    """Triggers are rendered as "When <trigger>"; drop a leading "when"/"if" the optimizer added."""
    t = trigger.strip().rstrip(".:")
    for prefix in ("when ", "if "):
        if t.lower().startswith(prefix):
            t = t[len(prefix):]
    return t


def playbook_text(lessons: list) -> str:
    rules = [l for l in _active(lessons) if l.target == "playbook"]
    if not rules:
        return ""
    lines = ["## Playbook: rules learned from past calls",
             "Each rule applies only in the situation it names and adds detail to the policies below. "
             "Safety always comes first: emergencies, privacy and no medical advice take priority over "
             "any playbook rule."]
    for l in rules:
        line = f"- [{l.id}] When {clean_trigger(l.trigger)}: {l.rule}"
        if l.example:
            line += f' For example, say: "{l.example}"'
        lines.append(line)
    return "\n".join(lines) + "\n"


def tool_notes(lessons: list) -> dict[str, list[str]]:
    notes: dict[str, list[str]] = {}
    for l in _active(lessons):
        if l.target.startswith("tool_desc:"):
            notes.setdefault(l.target.split(":", 1)[1], []).append(l.rule)
    return notes
