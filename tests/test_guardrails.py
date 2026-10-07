"""Every safety-critical rule is proven here without any LLM call."""

import pytest

from agent.ehr import MockEHR
from agent.session import Session
from agent.tools import run_tool


@pytest.fixture
def ehr():
    return MockEHR()


def open_slot(ehr, specialty="Primary Care"):
    res = run_tool("search_slots", {"specialty": specialty, "provider_name": None,
                                    "date_from": "2026-10-06", "date_to": "2026-10-16",
                                    "time_of_day": None}, Session(), ehr)
    return res["slots"][0]["slot_id"]


def verified(ehr, name="Marisol Quintero", dob="1985-03-14", rel="self", caller=None):
    s = Session()
    r = run_tool("verify_patient", {"full_name": name, "dob": dob, "caller_relationship": rel,
                                    "caller_name": caller}, s, ehr)
    assert r["ok"], r
    return s


def test_search_needs_no_verification_and_returns_no_phi(ehr):
    r = run_tool("search_slots", {"specialty": "Dental", "provider_name": None, "date_from": "2026-10-06",
                                  "date_to": "2026-10-10", "time_of_day": "morning"}, Session(), ehr)
    assert r["ok"] and r["slots"]
    assert all(set(x) == {"slot_id", "provider", "specialty", "start", "spoken"} for x in r["slots"])


def test_unverified_writes_refused(ehr):
    s = Session(last_user_message="yes book it")
    sid = open_slot(ehr)
    assert run_tool("hold_slot", {"slot_id": sid}, s, ehr)["error"] == "not_verified"
    assert run_tool("confirm_booking", {"patient_confirmation_quote": "yes"}, s, ehr)["error"] == "not_verified"
    assert run_tool("cancel_appointment", {"appointment_id": "A-101", "patient_confirmation_quote": "yes"},
                    s, ehr)["error"] == "not_verified"
    assert run_tool("list_appointments", {}, s, ehr)["error"] == "not_verified"


def test_booking_requires_verbatim_consent(ehr):
    s = verified(ehr)
    sid = open_slot(ehr)
    assert run_tool("hold_slot", {"slot_id": sid}, s, ehr)["ok"]
    s.last_user_message = "Hmm, what other times do you have?"
    r = run_tool("confirm_booking", {"patient_confirmation_quote": "yes please"}, s, ehr)
    assert r["error"] == "consent_not_found"
    assert ehr.get_slot(sid).status == "held"
    s.last_user_message = "Yes please, book that one."
    r = run_tool("confirm_booking", {"patient_confirmation_quote": "yes please"}, s, ehr)
    assert r["ok"] and ehr.get_appointment(r["appointment_id"]).patient_id == "P-001"


def test_patient_id_cannot_be_supplied_by_model(ehr):
    s = verified(ehr)
    r = run_tool("list_appointments", {"patient_id": "P-004"}, s, ehr)
    assert r["error"] == "unexpected_fields"


def test_cannot_touch_another_patients_appointment(ehr):
    s = verified(ehr)  # Marisol; A-103 belongs to Daniel Kim
    s.last_user_message = "yes cancel it"
    r = run_tool("cancel_appointment", {"appointment_id": "A-103", "patient_confirmation_quote": "yes cancel it"},
                 s, ehr)
    assert r["error"] == "appointment_not_found"
    assert ehr.get_appointment("A-103").status == "booked"


def test_urgent_escalation_locks_all_writes(ehr):
    s = verified(ehr)
    sid = open_slot(ehr)
    run_tool("hold_slot", {"slot_id": sid}, s, ehr)
    run_tool("escalate_to_human", {"reason": "chest pain", "urgency": "urgent"}, s, ehr)
    assert s.emergency_locked and ehr.get_slot(sid).status == "open"
    s.last_user_message = "yes book it"
    assert run_tool("hold_slot", {"slot_id": sid}, s, ehr)["error"] == "session_locked"
    assert run_tool("confirm_booking", {"patient_confirmation_quote": "yes book it"}, s, ehr)["error"] == "session_locked"


def test_verification_lockout_after_three_failures(ehr):
    s = Session()
    for _ in range(3):
        r = run_tool("verify_patient", {"full_name": "Daniel Kim", "dob": "1979-01-31",
                                        "caller_relationship": "self", "caller_name": None}, s, ehr)
        assert r["error"] == "no_match" and "1979" not in r["message"]
    r = run_tool("verify_patient", {"full_name": "Daniel Kim", "dob": "1979-01-30",
                                    "caller_relationship": "self", "caller_name": None}, s, ehr)
    assert r["error"] == "verification_locked" and s.verified_patient_id is None


def test_authorized_proxy_can_view_unlisted_proxy_cannot(ehr):
    s = verified(ehr, "James Whitfield", "1948-07-22", "daughter", "Sarah Whitfield")
    assert s.proxy_authorized and run_tool("list_appointments", {}, s, ehr)["ok"]
    s2 = verified(ehr, "Robert Chen", "1960-05-09", "wife", "Linda Chen")
    assert not s2.proxy_authorized
    assert run_tool("list_appointments", {}, s2, ehr)["error"] == "proxy_not_authorized"
    s2.last_user_message = "yes cancel"
    assert run_tool("cancel_appointment", {"appointment_id": "A-104", "patient_confirmation_quote": "yes cancel"},
                    s2, ehr)["error"] == "proxy_not_authorized"


def test_unlisted_proxy_may_book_new(ehr):
    s = verified(ehr, "Robert Chen", "1960-05-09", "wife", "Linda Chen")
    sid = open_slot(ehr, "Orthopedics")
    assert run_tool("hold_slot", {"slot_id": sid}, s, ehr)["ok"]
    s.last_user_message = "Yes, that works"
    r = run_tool("confirm_booking", {"patient_confirmation_quote": "yes, that works"}, s, ehr)
    assert r["ok"] and ehr.get_appointment(r["appointment_id"]).booked_by == "proxy"


REG = {"full_name": "Ananya Sharma", "dob": "1990-03-04", "phone": "617-555-0199", "email": None,
       "caller_relationship": "self", "caller_name": None, "patient_confirmation_quote": "yes that's right"}


def test_registration_requires_consent_and_creates_verified_patient(ehr):
    s = Session(last_user_message="wait, what?")
    assert run_tool("register_patient", REG, s, ehr)["error"] == "consent_not_found"
    s.last_user_message = "Yes that's right."
    r = run_tool("register_patient", REG, s, ehr)
    assert r["ok"] and s.verified_via == "registration"
    assert ehr.get_patient(s.verified_patient_id).created_via == "agent"


def test_registration_refuses_duplicates(ehr):
    s = Session(last_user_message="yes that's right")
    r = run_tool("register_patient", {**REG, "full_name": "emily novak", "dob": "1990-06-25"}, s, ehr)
    assert r["error"] == "existing_record"
    assert sum(p.full_name == "Emily Novak" for p in ehr.patients.values()) == 1


def test_registration_refuses_extra_fields_and_bad_dob(ehr):
    s = Session(last_user_message="yes that's right")
    assert run_tool("register_patient", {**REG, "ssn": "123-45-6789"}, s, ehr)["error"] == "unexpected_fields"
    assert run_tool("register_patient", {**REG, "dob": "2030-01-01"}, s, ehr)["error"] == "invalid_dob"


def test_slot_taken_externally_fails_cleanly(ehr):
    s = verified(ehr)
    sid = open_slot(ehr)
    run_tool("hold_slot", {"slot_id": sid}, s, ehr)
    ehr.external_book(sid)
    s.last_user_message = "yes book it"
    r = run_tool("confirm_booking", {"patient_confirmation_quote": "yes book it"}, s, ehr)
    assert r["error"] == "slot_unavailable"
    assert not [a for a in ehr.appointments.values() if a.slot_id == sid]


def test_late_cancellation_flagged(ehr):
    s = verified(ehr, "Priya Raman", "1992-11-02")
    s.last_user_message = "Yes, cancel it."
    r = run_tool("cancel_appointment", {"appointment_id": "A-102", "patient_confirmation_quote": "yes, cancel it"},
                 s, ehr)
    assert r["ok"] and r["late_cancellation"] is True


def test_reschedule_moves_appointment(ehr):
    s = verified(ehr)
    sid = open_slot(ehr)
    s.last_user_message = "yes move it"
    r = run_tool("reschedule_appointment", {"appointment_id": "A-101", "new_slot_id": sid,
                                            "patient_confirmation_quote": "yes move it"}, s, ehr)
    assert r["ok"]
    assert ehr.get_appointment("A-101").status == "cancelled"
    assert ehr.get_slot(sid).status == "booked"
