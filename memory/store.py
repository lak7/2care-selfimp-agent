"""Lessons memory: structured, auditable, reversible improvements learned from failed runs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

LESSONS_PATH = Path(__file__).resolve().parent / "lessons.json"

Status = Literal["candidate", "accepted", "rejected", "retired", "needs_human"]


class LessonSource(BaseModel):
    scenario: str
    run_id: str
    transcript: str = ""


class GateRecord(BaseModel):
    baseline_run: str = ""
    candidate_run: str = ""
    target_before: float = 0.0
    target_after: float = 0.0
    regressions: list[str] = Field(default_factory=list)
    new_hard_gates: list[str] = Field(default_factory=list)
    holdout_delta: float = 0.0
    passed: bool = False
    reason: str = ""


class Lesson(BaseModel):
    id: str
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    source: LessonSource
    violated: list[str]
    evidence: str
    root_cause: str
    target: str  # "playbook" | "tool_desc:<tool>" | "needs_code"
    trigger: str
    rule: str
    status: Status = "candidate"
    gate: GateRecord | None = None
    merged_from: list[str] = Field(default_factory=list)


class LessonStore:
    def __init__(self, path: Path = LESSONS_PATH):
        self.path = path
        self.lessons: list[Lesson] = []
        if path.exists() and path.read_text().strip():
            self.lessons = [Lesson(**d) for d in json.loads(path.read_text())]

    def save(self) -> None:
        self.path.write_text(json.dumps([l.model_dump(mode="json") for l in self.lessons], indent=2))

    def next_id(self) -> str:
        n = max((int(l.id.split("-")[1]) for l in self.lessons), default=0) + 1
        return f"L-{n:03d}"

    def get(self, lesson_id: str) -> Lesson:
        for l in self.lessons:
            if l.id == lesson_id:
                return l
        raise KeyError(lesson_id)

    def accepted(self) -> list[Lesson]:
        return [l for l in self.lessons if l.status == "accepted"]

    def add(self, lesson: Lesson) -> Lesson:
        self.lessons.append(lesson)
        return lesson

    def set_status(self, lesson_id: str, status: Status) -> None:
        self.get(lesson_id).status = status

    def merged(self, base_id: str, new: Lesson) -> Lesson:
        """A new candidate that replaces `base_id` once gated (dedupe instead of piling up rules)."""
        new.merged_from = [base_id]
        return new
