"""Clinic storage behind an EHR adapter boundary.

Tools never touch storage directly; they go through `EHRAdapter`. `MockEHR` is an
in-memory implementation seeded from data/clinic_seed.json. A real integration
(Epic, athenahealth, Cliniko...) would be another adapter with the same surface.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, Field

SEED_PATH = Path(__file__).resolve().parent.parent / "data" / "clinic_seed.json"


class Provider(BaseModel):
    id: str
    name: str
    specialty: str
    weekdays: list[int]


class Slot(BaseModel):
    id: str
    provider_id: str
    start: datetime
    end: datetime
    status: Literal["open", "held", "booked"] = "open"


class Proxy(BaseModel):
    name: str
    relationship: str


class Patient(BaseModel):
    id: str
    full_name: str
    dob: date
    phone: str
    email: str | None = None
    created_via: Literal["seed", "agent"] = "seed"
    authorized_proxies: list[Proxy] = Field(default_factory=list)


class Appointment(BaseModel):
    id: str
    patient_id: str
    slot_id: str
    status: Literal["booked", "cancelled"] = "booked"
    booked_by: Literal["seed", "self", "proxy"] = "seed"


class WaitlistEntry(BaseModel):
    patient_id: str
    specialty: str
    date_from: date
    date_to: date


def normalize_name(name: str) -> str:
    return " ".join("".join(c for c in name.lower() if c.isalnum() or c.isspace()).split())


def slot_id_for(provider_id: str, start: datetime) -> str:
    return f"S-{provider_id}-{start:%Y%m%d-%H%M}"


class EHRAdapter(Protocol):
    now: datetime

    def find_patient(self, full_name: str, dob: date) -> Patient | None: ...
    def get_patient(self, patient_id: str) -> Patient: ...
    def create_patient(self, full_name: str, dob: date, phone: str, email: str | None,
                       proxies: list[Proxy]) -> Patient: ...
    def search_slots(self, specialty: str | None, provider_id: str | None, date_from: date,
                     date_to: date, time_of_day: str | None, limit: int = 6) -> list[Slot]: ...
    def get_slot(self, slot_id: str) -> Slot | None: ...
    def hold(self, slot_id: str) -> bool: ...
    def release(self, slot_id: str) -> None: ...
    def book(self, patient_id: str, slot_id: str, booked_by: str) -> Appointment: ...
    def cancel(self, appointment_id: str) -> Appointment: ...
    def get_appointment(self, appointment_id: str) -> Appointment | None: ...
    def appointments_for(self, patient_id: str) -> list[Appointment]: ...
    def add_waitlist(self, entry: WaitlistEntry) -> None: ...


class MockEHR:
    def __init__(self, seed: dict | None = None, patch: list[dict] | None = None):
        seed = copy.deepcopy(seed or json.loads(SEED_PATH.read_text()))
        self.clinic = seed["clinic"]
        self.now = datetime.fromisoformat(self.clinic["today"])
        self.providers = {p["id"]: Provider(**p) for p in seed["providers"]}
        self.patients = {p["id"]: Patient(**p) for p in seed["patients"]}
        self.slots: dict[str, Slot] = {}
        self._generate_slots(seed["schedule"])
        self.appointments: dict[str, Appointment] = {}
        for a in seed["appointments"]:
            appt = Appointment(**a)
            self.appointments[appt.id] = appt
            self.slots[appt.slot_id].status = "booked"
        self.waitlist: list[WaitlistEntry] = []
        self._next_appt = 900
        self._next_patient = 900
        for op in patch or []:
            self._apply_patch(op)

    # --- seeding -----------------------------------------------------------------
    def _generate_slots(self, sched: dict) -> None:
        step = timedelta(minutes=sched["slot_minutes"])
        ratio = int(sched["prebooked_ratio"] * 100)
        for offset in range(sched["days_ahead"]):
            day = self.now.date() + timedelta(days=offset)
            for prov in self.providers.values():
                if day.weekday() not in prov.weekdays:
                    continue
                for start_s, end_s in sched["blocks"]:
                    t = datetime.combine(day, datetime.strptime(start_s, "%H:%M").time())
                    end = datetime.combine(day, datetime.strptime(end_s, "%H:%M").time())
                    while t + step <= end:
                        sid = slot_id_for(prov.id, t)
                        # Deterministic "other patients" bookings so availability looks real.
                        h = int(hashlib.sha256(sid.encode()).hexdigest(), 16) % 100
                        status = "booked" if h < ratio else "open"
                        self.slots[sid] = Slot(id=sid, provider_id=prov.id, start=t, end=t + step,
                                               status=status)
                        t += step

    def _apply_patch(self, op: dict) -> None:
        kind = op["op"]
        if kind == "block_slots":
            for s in self._slots_matching(op):
                s.status = "booked"
        elif kind == "open_slots":
            for sid in op["slot_ids"]:
                self.slots[sid].status = "open"
        elif kind == "add_patient":
            p = Patient(**op["patient"])
            self.patients[p.id] = p
        elif kind == "add_appointment":
            appt = Appointment(**op["appointment"])
            self.appointments[appt.id] = appt
            self.slots[appt.slot_id].status = "booked"
        else:
            raise ValueError(f"unknown patch op {kind}")

    def _slots_matching(self, op: dict) -> list[Slot]:
        d_from = date.fromisoformat(op.get("date_from", "1900-01-01"))
        d_to = date.fromisoformat(op.get("date_to", "2999-01-01"))
        out = []
        for s in self.slots.values():
            prov = self.providers[s.provider_id]
            if op.get("provider_id") and prov.id != op["provider_id"]:
                continue
            if op.get("specialty") and prov.specialty.lower() != op["specialty"].lower():
                continue
            if d_from <= s.start.date() <= d_to:
                out.append(s)
        return out

    # --- adapter surface -----------------------------------------------------------
    def find_patient(self, full_name: str, dob: date) -> Patient | None:
        target = normalize_name(full_name)
        for p in self.patients.values():
            if normalize_name(p.full_name) == target and p.dob == dob:
                return p
        return None

    def get_patient(self, patient_id: str) -> Patient:
        return self.patients[patient_id]

    def create_patient(self, full_name: str, dob: date, phone: str, email: str | None,
                       proxies: list[Proxy]) -> Patient:
        self._next_patient += 1
        p = Patient(id=f"P-{self._next_patient}", full_name=full_name.strip(), dob=dob, phone=phone,
                    email=email, created_via="agent", authorized_proxies=proxies)
        self.patients[p.id] = p
        return p

    def search_slots(self, specialty, provider_id, date_from, date_to, time_of_day, limit=6):
        out = []
        for s in sorted(self.slots.values(), key=lambda s: s.start):
            if s.status != "open" or s.start <= self.now:
                continue
            prov = self.providers[s.provider_id]
            if specialty and prov.specialty.lower() != specialty.lower():
                continue
            if provider_id and prov.id != provider_id:
                continue
            if not (date_from <= s.start.date() <= date_to):
                continue
            if time_of_day == "morning" and s.start.hour >= 12:
                continue
            if time_of_day == "afternoon" and s.start.hour < 12:
                continue
            out.append(s)
        # Spread results across days rather than returning one packed morning.
        by_day: dict[date, list[Slot]] = {}
        for s in out:
            by_day.setdefault(s.start.date(), []).append(s)
        picked: list[Slot] = []
        while len(picked) < limit and any(by_day.values()):
            for d in sorted(by_day):
                if by_day[d] and len(picked) < limit:
                    picked.append(by_day[d].pop(0))
        return sorted(picked, key=lambda s: s.start)

    def get_slot(self, slot_id: str) -> Slot | None:
        return self.slots.get(slot_id)

    def hold(self, slot_id: str) -> bool:
        s = self.slots.get(slot_id)
        if not s or s.status != "open" or s.start <= self.now:
            return False
        s.status = "held"
        return True

    def release(self, slot_id: str) -> None:
        s = self.slots.get(slot_id)
        if s and s.status == "held":
            s.status = "open"

    def book(self, patient_id: str, slot_id: str, booked_by: str) -> Appointment:
        s = self.slots[slot_id]
        if s.status == "booked":
            raise ValueError("slot_unavailable")
        s.status = "booked"
        self._next_appt += 1
        appt = Appointment(id=f"A-{self._next_appt}", patient_id=patient_id, slot_id=slot_id,
                           booked_by=booked_by)
        self.appointments[appt.id] = appt
        return appt

    def cancel(self, appointment_id: str) -> Appointment:
        appt = self.appointments[appointment_id]
        appt.status = "cancelled"
        self.slots[appt.slot_id].status = "open"
        return appt

    def get_appointment(self, appointment_id: str) -> Appointment | None:
        return self.appointments.get(appointment_id)

    def appointments_for(self, patient_id: str) -> list[Appointment]:
        return [a for a in self.appointments.values()
                if a.patient_id == patient_id and a.status == "booked"]

    def add_waitlist(self, entry: WaitlistEntry) -> None:
        self.waitlist.append(entry)

    # --- harness helpers (not exposed to the agent) -------------------------------
    def external_book(self, slot_id: str) -> None:
        """World event: someone else books this slot (even if our session holds it)."""
        self.slots[slot_id].status = "booked"

    def describe_slot(self, slot_id: str) -> dict:
        s = self.slots[slot_id]
        prov = self.providers[s.provider_id]
        return {"slot_id": s.id, "provider": prov.name, "specialty": prov.specialty,
                "start": s.start.isoformat(timespec="minutes"),
                "spoken": spoken_time(s.start)}

    def snapshot(self) -> dict:
        return {
            "patients": {pid: p.model_dump(mode="json") for pid, p in self.patients.items()},
            "appointments": {aid: a.model_dump(mode="json") for aid, a in self.appointments.items()},
            "waitlist": [w.model_dump(mode="json") for w in self.waitlist],
            "slots": {sid: s.status for sid, s in self.slots.items()},
        }


def spoken_time(dt: datetime) -> str:
    """'Tuesday, October 13 at 10:00 AM' — how a receptionist would say it."""
    hour = dt.strftime("%I").lstrip("0")
    return f"{dt:%A}, {dt:%B} {dt.day} at {hour}:{dt:%M} {dt:%p}"
