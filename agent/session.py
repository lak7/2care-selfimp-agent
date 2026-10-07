"""Structured conversation state, owned by code (never by the model)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

MAX_VERIFY_ATTEMPTS = 3


@dataclass
class Session:
    caller_role: Literal["unknown", "self", "proxy"] = "unknown"
    caller_name: str | None = None
    caller_relationship: str | None = None
    verified_patient_id: str | None = None
    verified_via: Literal["verification", "registration"] | None = None
    proxy_authorized: bool = False
    failed_verifications: int = 0
    held_slot_id: str | None = None
    emergency_locked: bool = False
    escalations: list[dict] = field(default_factory=list)
    last_user_message: str = ""

    @property
    def verification_locked(self) -> bool:
        return self.failed_verifications >= MAX_VERIFY_ATTEMPTS

    def to_dict(self) -> dict:
        return asdict(self) | {"verification_locked": self.verification_locked}
