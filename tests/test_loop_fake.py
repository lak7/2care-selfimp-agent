"""End-to-end agent loop with a scripted model: proves wiring, trace and state for $0."""

from agent.ehr import MockEHR
from agent.loop import Agent
from llm import FakeLLM, fake_call, fake_text


def scripted(steps):
    it = iter(steps)
    return lambda input, tools: next(it)


def test_booking_flow_through_loop():
    ehr = MockEHR()
    slot = ehr.search_slots("Dental", None, ehr.now.date(), ehr.now.date().replace(day=16), None)[0]
    llm = FakeLLM({"agent": scripted([
        fake_call("verify_patient", {"full_name": "Maria Gonzalez", "dob": "1985-03-14",
                                     "caller_relationship": "self", "caller_name": None}, "c1"),
        fake_call("hold_slot", {"slot_id": slot.id}, "c2"),
        fake_text("I have that held. Shall I book it?"),
        fake_call("confirm_booking", {"patient_confirmation_quote": "yes book it"}, "c3"),
        fake_text("You're all set."),
    ])})
    agent = Agent(ehr, llm)
    assert agent.step("Hi, Maria Gonzalez, March 14 1985, I need a dental visit") == \
        "I have that held. Shall I book it?"
    assert agent.step("Yes book it please") == "You're all set."
    assert [t["tool"] for t in agent.trace] == ["verify_patient", "hold_slot", "confirm_booking"]
    assert all(t["result"]["ok"] for t in agent.trace)
    booked = [a for a in ehr.appointments.values() if a.slot_id == slot.id]
    assert booked and booked[0].patient_id == "P-001"
    # tool outputs were fed back to the model
    assert any(i.get("type") == "function_call_output" for i in llm.calls[-1][1])


def test_playbook_compiles_into_prompt():
    from memory.store import Lesson, LessonSource
    l = Lesson(id="L-001", source=LessonSource(scenario="T07", run_id="r"), violated=["HG1"],
               evidence="e", root_cause="r", target="playbook", trigger="a caller books for someone else",
               rule="Verify the patient's details, not the caller's.", status="accepted")
    t = Lesson(**{**l.model_dump(), "id": "L-002", "target": "tool_desc:verify_patient",
                  "rule": "Use the patient's name."})
    agent = Agent(MockEHR(), FakeLLM({}), [l, t])
    assert "[L-001] When a caller books for someone else: Verify" in agent.instructions
    verify = next(x for x in agent.tools if x["name"] == "verify_patient")
    assert verify["description"].endswith("Note: Use the patient's name.")
