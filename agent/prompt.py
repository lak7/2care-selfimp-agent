"""System prompt assembly: the learned Playbook compiled from lessons + the fixed base sections.

The Playbook goes FIRST: in testing, gpt-5-nano largely ignored rules appended after the base
prompt. Costs a little prompt caching once lessons exist; worth it.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from agent.ehr import MockEHR, spoken_time
from agent.tools import tool_schemas
from memory.compile import playbook_text, tool_notes

BASE = Path(__file__).resolve().parent / "prompts" / "base_v0.md"


def build_system_prompt(ehr: MockEHR, lessons: list) -> str:
    now: datetime = ehr.now
    providers = "\n".join(f"- {p.name}, {p.specialty}" for p in ehr.providers.values())
    text = BASE.read_text().format(
        clinic_name=ehr.clinic["name"], now_spoken=spoken_time(now), timezone=ehr.clinic["timezone"],
        today_iso=now.date().isoformat(), providers=providers)
    playbook = playbook_text(lessons)
    return (playbook + "\n" if playbook else "") + text


def build_tools(lessons: list) -> list[dict]:
    return tool_schemas(tool_notes(lessons))


def prompt_hash(ehr: MockEHR, lessons: list) -> str:
    blob = build_system_prompt(ehr, lessons) + json.dumps(build_tools(lessons), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]
