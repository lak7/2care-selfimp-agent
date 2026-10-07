"""Agent tools with code-enforced guardrails.

Design rule: the prompt *asks* for safe behaviour; these functions *guarantee* it.
- patient_id is never a model-supplied argument; it comes from the verified Session.
- Every write needs: a verified patient, no emergency lock, and a consent quote that
  appears verbatim in the patient's latest message.
- Errors are returned as data ({"ok": false, ...}) so the model can recover.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Callable

from agent.ehr import EHRAdapter, Proxy, WaitlistEntry, normalize_name
from agent.session import MAX_VERIFY_ATTEMPTS, Session

WRITE_TOOLS = {"register_patient", "confirm_booking", "reschedule_appointment",
               "cancel_appointment", "join_waitlist"}
SPECIALTIES = ["Primary Care", "Dental", "Orthopedics", "Dermatology"]


class ToolError(Exception):
    def __init__(self, code: str, message: str):
        self.code, self.message = code, message


def err(code: str, message: str) -> dict:
    return {"ok": False, "error": code, "message": message}


# --- guardrails -------------------------------------------------------------------

def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def check_consent(quote: str, last_user_message: str) -> None:
    q = _norm(quote or "")
    if len(q) < 2 or q not in _norm(last_user_message):
        raise ToolError("consent_not_found",
                        "patient_confirmation_quote must be copied verbatim from the patient's most "
                        "recent message, and that message must explicitly agree to this action. "
                        "Read the details back and ask the patient to confirm first.")


def require_verified(s: Session) -> str:
    if not s.verified_patient_id:
        raise ToolError("not_verified", "Verify the patient's identity (verify_patient) or register "
                                        "them (register_patient) first.")
    return s.verified_patient_id


def require_not_locked(s: Session) -> None:
    if s.emergency_locked:
        raise ToolError("session_locked", "This session was escalated as urgent. No scheduling "
                                          "changes are allowed; keep directing the caller to "
                                          "emergency care.")


def require_proxy_access(s: Session) -> None:
    if s.caller_role == "proxy" and not s.proxy_authorized:
        raise ToolError("proxy_not_authorized",
                        "The caller is not an authorized proxy on this patient's record. They may "
                        "book a NEW appointment, but cannot view or change existing appointments. "
                        "Do not reveal whether appointments exist.")


def _parse_date(value: str, field: str) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        raise ToolError("invalid_date", f"{field} must be YYYY-MM-DD, got {value!r}.")


def _owned_appointment(ehr: EHRAdapter, s: Session, appointment_id: str):
    appt = ehr.get_appointment(appointment_id)
    if not appt or appt.patient_id != s.verified_patient_id or appt.status != "booked":
        raise ToolError("appointment_not_found", "No active appointment with that id for this patient.")
    return appt


def _slot_view(ehr: EHRAdapter, slot_id: str) -> dict:
    return ehr.describe_slot(slot_id)


# --- tools --------------------------------------------------------------------------

def verify_patient(s: Session, ehr: EHRAdapter, full_name: str, dob: str,
                   caller_relationship: str, caller_name: str | None) -> dict:
    if s.verification_locked:
        raise ToolError("verification_locked", "Too many failed attempts. Offer a transfer to staff "
                                               "(escalate_to_human, routine). Share no information.")
    d = _parse_date(dob, "dob")
    patient = ehr.find_patient(full_name, d)
    if not patient:
        s.failed_verifications += 1
        left = MAX_VERIFY_ATTEMPTS - s.failed_verifications
        return err("no_match", f"No patient matches that name and date of birth. "
                               f"{left} attempt(s) remaining. Do not hint which detail was wrong.")
    is_self = caller_relationship.strip().lower() in {"self", "me", "patient"}
    if s.held_slot_id:
        ehr.release(s.held_slot_id)
        s.held_slot_id = None
    s.verified_patient_id, s.verified_via = patient.id, "verification"
    s.caller_role = "self" if is_self else "proxy"
    s.caller_relationship = None if is_self else caller_relationship
    s.caller_name = caller_name
    s.proxy_authorized = (not is_self) and bool(caller_name) and any(
        normalize_name(p.name) == normalize_name(caller_name) for p in patient.authorized_proxies)
    return {"ok": True, "patient_first_name": patient.full_name.split()[0],
            "caller_role": s.caller_role, "proxy_authorized": s.proxy_authorized}


def register_patient(s: Session, ehr: EHRAdapter, full_name: str, dob: str, phone: str,
                     email: str | None, caller_relationship: str, caller_name: str | None,
                     patient_confirmation_quote: str) -> dict:
    require_not_locked(s)
    if s.verified_patient_id:
        raise ToolError("already_verified", "A patient is already verified in this call.")
    d = _parse_date(dob, "dob")
    if not (date(1900, 1, 1) <= d < ehr.now.date()):
        raise ToolError("invalid_dob", "Date of birth must be a real past date.")
    digits = re.sub(r"\D", "", phone or "")
    if not 10 <= len(digits) <= 15:
        raise ToolError("invalid_phone", "Phone number must have 10-15 digits.")
    if len(full_name.split()) < 2:
        raise ToolError("invalid_name", "Need the patient's first and last name.")
    if ehr.find_patient(full_name, d):
        raise ToolError("existing_record", "A patient with this name and date of birth already "
                                           "exists. Do not create a new record; verify them with "
                                           "verify_patient instead.")
    check_consent(patient_confirmation_quote, s.last_user_message)
    is_self = caller_relationship.strip().lower() in {"self", "me", "patient"}
    proxies = [] if is_self or not caller_name else [Proxy(name=caller_name,
                                                           relationship=caller_relationship)]
    p = ehr.create_patient(full_name, d, phone, email, proxies)
    s.verified_patient_id, s.verified_via = p.id, "registration"
    s.caller_role = "self" if is_self else "proxy"
    s.caller_name, s.proxy_authorized = caller_name, not is_self
    return {"ok": True, "patient_id_created": True, "patient_first_name": full_name.split()[0]}


def search_slots(s: Session, ehr: EHRAdapter, specialty: str | None, provider_name: str | None,
                 date_from: str, date_to: str, time_of_day: str | None) -> dict:
    d_from, d_to = _parse_date(date_from, "date_from"), _parse_date(date_to, "date_to")
    today = ehr.now.date()
    if d_from < today:
        d_from = today
    if d_to < d_from:
        raise ToolError("invalid_range", "date_to is before date_from.")
    provider_id = None
    if provider_name:
        target = normalize_name(provider_name.replace("Dr.", "").replace("Dr ", ""))
        matches = [p for p in ehr.providers.values() if target in normalize_name(p.name)]
        if not matches:
            raise ToolError("unknown_provider", f"No provider matches {provider_name!r}.")
        provider_id = matches[0].id
    if specialty and specialty.lower() not in {x.lower() for x in SPECIALTIES}:
        raise ToolError("unknown_specialty", f"Specialties offered: {', '.join(SPECIALTIES)}.")
    slots = ehr.search_slots(specialty, provider_id, d_from, d_to, time_of_day)
    horizon = today + timedelta(days=13)
    return {"ok": True, "slots": [_slot_view(ehr, x.id) for x in slots],
            "note": ("No open slots match. Offer nearby alternatives or the waitlist."
                     if not slots else "Only quote times from this list."),
            "schedule_open_until": horizon.isoformat()}


def hold_slot(s: Session, ehr: EHRAdapter, slot_id: str) -> dict:
    require_verified(s)
    require_not_locked(s)
    if s.held_slot_id and s.held_slot_id != slot_id:
        ehr.release(s.held_slot_id)
        s.held_slot_id = None
    if s.held_slot_id == slot_id:
        return {"ok": True, "held": _slot_view(ehr, slot_id)}
    if not ehr.hold(slot_id):
        raise ToolError("slot_unavailable", "That slot is no longer available. Search again.")
    s.held_slot_id = slot_id
    return {"ok": True, "held": _slot_view(ehr, slot_id),
            "next": "Read the details back and get an explicit yes before confirm_booking."}


def confirm_booking(s: Session, ehr: EHRAdapter, patient_confirmation_quote: str) -> dict:
    pid = require_verified(s)
    require_not_locked(s)
    if not s.held_slot_id:
        raise ToolError("no_held_slot", "Hold a slot first with hold_slot.")
    check_consent(patient_confirmation_quote, s.last_user_message)
    slot_id = s.held_slot_id
    try:
        appt = ehr.book(pid, slot_id, booked_by="proxy" if s.caller_role == "proxy" else "self")
    except ValueError:
        s.held_slot_id = None
        raise ToolError("slot_unavailable", "That slot was just taken by someone else. Apologize, "
                                            "search again and offer alternatives. Nothing was booked.")
    s.held_slot_id = None
    return {"ok": True, "appointment_id": appt.id, **_slot_view(ehr, slot_id)}


def list_appointments(s: Session, ehr: EHRAdapter) -> dict:
    pid = require_verified(s)
    require_proxy_access(s)
    return {"ok": True, "appointments": [
        {"appointment_id": a.id, **_slot_view(ehr, a.slot_id)} for a in ehr.appointments_for(pid)]}


def reschedule_appointment(s: Session, ehr: EHRAdapter, appointment_id: str, new_slot_id: str,
                           patient_confirmation_quote: str) -> dict:
    require_verified(s)
    require_not_locked(s)
    require_proxy_access(s)
    old = _owned_appointment(ehr, s, appointment_id)
    check_consent(patient_confirmation_quote, s.last_user_message)
    slot = ehr.get_slot(new_slot_id)
    if not slot or slot.start <= ehr.now or (slot.status != "open" and s.held_slot_id != new_slot_id):
        raise ToolError("slot_unavailable", "The new slot is not available. Search again.")
    if s.held_slot_id == new_slot_id:
        ehr.release(new_slot_id)
        s.held_slot_id = None
    new = ehr.book(s.verified_patient_id, new_slot_id,
                   booked_by="proxy" if s.caller_role == "proxy" else "self")
    ehr.cancel(old.id)
    return {"ok": True, "cancelled_appointment_id": old.id, "new_appointment_id": new.id,
            **_slot_view(ehr, new_slot_id)}


def cancel_appointment(s: Session, ehr: EHRAdapter, appointment_id: str,
                       patient_confirmation_quote: str) -> dict:
    require_verified(s)
    require_not_locked(s)
    require_proxy_access(s)
    appt = _owned_appointment(ehr, s, appointment_id)
    check_consent(patient_confirmation_quote, s.last_user_message)
    late = ehr.get_slot(appt.slot_id).start - ehr.now < timedelta(hours=24)
    ehr.cancel(appt.id)
    return {"ok": True, "cancelled_appointment_id": appt.id, "late_cancellation": late}


def join_waitlist(s: Session, ehr: EHRAdapter, specialty: str, date_from: str, date_to: str) -> dict:
    pid = require_verified(s)
    require_not_locked(s)
    ehr.add_waitlist(WaitlistEntry(patient_id=pid, specialty=specialty,
                                   date_from=_parse_date(date_from, "date_from"),
                                   date_to=_parse_date(date_to, "date_to")))
    return {"ok": True, "waitlisted": True}


def escalate_to_human(s: Session, ehr: EHRAdapter, reason: str, urgency: str) -> dict:
    ticket = f"ESC-{len(s.escalations) + 1}"
    s.escalations.append({"ticket": ticket, "reason": reason, "urgency": urgency})
    if urgency == "urgent":
        s.emergency_locked = True
        if s.held_slot_id:
            ehr.release(s.held_slot_id)
            s.held_slot_id = None
    return {"ok": True, "ticket": ticket,
            "message": "On-call clinical staff notified." if urgency == "urgent"
            else "Front-desk staff will follow up with the caller."}


# --- registry & schemas ------------------------------------------------------------

S = {"type": "string"}
NS = {"type": ["string", "null"]}
QUOTE = {"type": "string", "description": "Exact words copied from the patient's latest message "
                                          "that explicitly agree to this action."}

TOOLS: dict[str, tuple[Callable, dict]] = {
    "verify_patient": (verify_patient, {
        "full_name": {**S, "description": "The PATIENT's full name (not the caller's, if different)."},
        "dob": {**S, "description": "Patient date of birth, YYYY-MM-DD."},
        "caller_relationship": {**S, "description": "'self' if the caller is the patient, otherwise "
                                                     "their relationship, e.g. 'daughter'."},
        "caller_name": {**NS, "description": "Caller's full name when calling for someone else."}}),
    "register_patient": (register_patient, {
        "full_name": S, "dob": {**S, "description": "YYYY-MM-DD"}, "phone": S, "email": NS,
        "caller_relationship": S, "caller_name": NS, "patient_confirmation_quote": QUOTE}),
    "search_slots": (search_slots, {
        "specialty": {"type": ["string", "null"], "enum": SPECIALTIES + [None]},
        "provider_name": NS,
        "date_from": {**S, "description": "YYYY-MM-DD"},
        "date_to": {**S, "description": "YYYY-MM-DD"},
        "time_of_day": {"type": ["string", "null"], "enum": ["morning", "afternoon", None]}}),
    "hold_slot": (hold_slot, {"slot_id": S}),
    "confirm_booking": (confirm_booking, {"patient_confirmation_quote": QUOTE}),
    "list_appointments": (list_appointments, {}),
    "reschedule_appointment": (reschedule_appointment, {
        "appointment_id": S, "new_slot_id": S, "patient_confirmation_quote": QUOTE}),
    "cancel_appointment": (cancel_appointment, {"appointment_id": S, "patient_confirmation_quote": QUOTE}),
    "join_waitlist": (join_waitlist, {
        "specialty": {"type": "string", "enum": SPECIALTIES},
        "date_from": {**S, "description": "YYYY-MM-DD"}, "date_to": {**S, "description": "YYYY-MM-DD"}}),
    "escalate_to_human": (escalate_to_human, {
        "reason": S, "urgency": {"type": "string", "enum": ["routine", "urgent"]}}),
}

BASE_DESCRIPTIONS: dict[str, str] = {
    "verify_patient": "Verify the patient's identity by full name and date of birth. Required before "
                      "any access to the patient's records or any scheduling change.",
    "register_patient": "Create a record for a NEW patient. Collect only name, date of birth, phone "
                        "and optional email. Read every detail back and get explicit confirmation first.",
    "search_slots": "Find open appointment slots. Returns slot ids and spoken times; these are the "
                    "only times you may offer.",
    "hold_slot": "Temporarily hold one slot for the verified patient while you confirm with them.",
    "confirm_booking": "Book the currently held slot for the verified patient. Requires the patient's "
                       "explicit agreement in their latest message.",
    "list_appointments": "List the verified patient's upcoming appointments.",
    "reschedule_appointment": "Move an existing appointment to a new open slot. Requires explicit agreement.",
    "cancel_appointment": "Cancel an existing appointment. Requires explicit agreement. Cancellations "
                          "under 24 hours ahead incur a late-cancellation fee; tell the patient first.",
    "join_waitlist": "Add the verified patient to the waitlist for a specialty and date range.",
    "escalate_to_human": "Hand off to clinic staff. Use urgency 'urgent' for any possible medical "
                         "emergency (this also locks scheduling), 'routine' otherwise.",
}


def tool_schemas(notes: dict[str, list[str]] | None = None) -> list[dict]:
    """Responses-API function tool definitions (strict). `notes` are learned lesson amendments."""
    notes = notes or {}
    out = []
    for name, (_, props) in TOOLS.items():
        desc = BASE_DESCRIPTIONS[name]
        for n in notes.get(name, []):
            desc += f" Note: {n}"
        out.append({"type": "function", "name": name, "description": desc, "strict": True,
                    "parameters": {"type": "object", "properties": props,
                                   "required": list(props), "additionalProperties": False}})
    return out


def run_tool(name: str, args: dict, session: Session, ehr: EHRAdapter) -> dict:
    if name not in TOOLS:
        return err("unknown_tool", f"No tool named {name}.")
    fn, props = TOOLS[name]
    unknown = set(args) - set(props)
    if unknown:  # minimum-necessary: refuse fields outside the schema
        return err("unexpected_fields", f"Fields not allowed: {sorted(unknown)}.")
    try:
        return fn(session, ehr, **{k: args.get(k) for k in props})
    except ToolError as e:
        return err(e.code, e.message)
    except TypeError as e:
        return err("bad_arguments", str(e))
