"""Scenario spec (YAML) — persona, world setup, events and expectations."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

SCENARIO_DIR = Path(__file__).resolve().parent / "scenarios"


class Persona(BaseModel):
    description: str
    goal: str
    hidden_facts: dict[str, str] = Field(default_factory=dict)
    opening_line: str
    style: str = "Cooperative, natural, brief."
    # Exact lines the patient says at a given patient turn (turn 1 = opening line). Used for the
    # events a scenario exists to test (an emergency, a change of mind), so they always happen
    # instead of being left to the simulator's discretion.
    script: dict[int, str] = Field(default_factory=dict)
    # Deterministic corrections: if the receptionist's reply matches `when` (regex, case-insensitive),
    # the patient says `say` next (once). A small simulator model often fails to notice a wrong
    # read-back, which would make the failure impossible for the agent to learn from.
    reactions: list[dict[str, str]] = Field(default_factory=list)


class Behaviors(BaseModel):
    asr_noise: bool = False
    asr_confusions: dict[str, str] = Field(default_factory=dict)  # extra word -> misheard word
    adversarial: bool = False


class WorldEvent(BaseModel):
    action: Literal["take_held_slot"]
    after_turn: int = 1


class Check(BaseModel):
    check: str
    args: dict = Field(default_factory=dict)


class JudgeItem(BaseModel):
    id: str
    text: str


class Expect(BaseModel):
    state: list[Check] = Field(default_factory=list)
    trace: list[Check] = Field(default_factory=list)
    judge: list[JudgeItem] = Field(default_factory=list)


class Scenario(BaseModel):
    id: str
    split: Literal["train", "holdout"]
    title: str
    patient_id: str | None = None          # the record this call is legitimately about (None = new/none)
    emergency: bool = False
    seed_patch: list[dict] = Field(default_factory=list)
    persona: Persona
    behaviors: Behaviors = Field(default_factory=Behaviors)
    world_events: list[WorldEvent] = Field(default_factory=list)
    max_turns: int = 10
    expect: Expect = Field(default_factory=Expect)


def load_scenarios(split: str = "all", only: list[str] | None = None,
                   profile: str | None = None) -> list[Scenario]:
    """`profile` (from config.yaml `profiles`) restricts to a named scenario set, e.g. the demo set."""
    if profile:
        from llm import config
        ids = config()["profiles"][profile]["scenarios"]
        if not only and ids != "all":  # an explicit --only wins over the profile's list
            only = ids
    out = []
    for path in sorted(SCENARIO_DIR.glob("*/*.yaml")):
        sc = Scenario(**yaml.safe_load(path.read_text()))
        if split != "all" and sc.split != split:
            continue
        if only and not any(sc.id.startswith(o) for o in only):
            continue
        out.append(sc)
    return out
