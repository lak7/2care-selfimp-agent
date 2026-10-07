"""Conversation driver behaviour: scripted events, the agent's last word, progress-free scoring ($0)."""

from evals.checks import hg_hallucination
from evals.runner import Result, run_scenario
from evals.scenario import Scenario, load_scenarios
from llm import FakeLLM, fake_text
from memory.compile import clean_trigger


def scenario(**persona):
    return Scenario(id="X", split="train", title="t",
                    persona={"description": "d", "goal": "g", "opening_line": "hello", **persona})


def agent_echo(input, tools):
    return fake_text(f"agent heard: {input[-1]['content']}")


def test_done_tail_gets_a_final_agent_turn():
    llm = FakeLLM({"agent": agent_echo, "simulator": lambda i, t: fake_text("Yes, please cancel it. [DONE]")})
    r = run_scenario(scenario(), [], llm, 0)
    assert r.transcript[-1] == {"role": "agent", "turn": 2, "text": "agent heard: Yes, please cancel it."}
    assert r.ended_by == "done"


def test_scripted_line_is_said_verbatim_and_beats_an_early_hangup():
    llm = FakeLLM({"agent": agent_echo, "simulator": lambda i, t: fake_text("Bye! [DONE]")})
    r = run_scenario(scenario(script={3: "I have chest pain."}), [], llm, 0)
    said = [m["said"] for m in r.transcript if m["role"] == "patient"]
    assert said[:2] == ["hello", "I have chest pain."]   # hang-up at turn 2 replaced by the scripted event
    assert any(m["text"] == "agent heard: I have chest pain." for m in r.transcript)


def test_current_time_is_not_a_hallucination():
    r = Result(scenario_id="X", split="train", run_idx=0, transcript=[
        {"role": "patient", "turn": 1, "said": "hi", "text": "hi"},
        {"role": "agent", "turn": 1, "text": "It's Monday, October 5 at 9:00 AM right now."}])
    assert hg_hallucination(r) == []


def test_clean_trigger_and_profiles():
    assert clean_trigger("When a caller books for someone else.") == "a caller books for someone else"
    demo = load_scenarios(profile="demo")
    assert 0 < len(demo) < len(load_scenarios())
    assert [s.id for s in load_scenarios(only=["T01"], profile="demo")] == ["T01_happy_path"]


def test_done_marker_is_case_insensitive():
    llm = FakeLLM({"agent": agent_echo, "simulator": lambda i, t: fake_text("Okay, bye. [Done]")})
    r = run_scenario(scenario(), [], llm, 0)
    assert r.ended_by == "done" and len([m for m in r.transcript if m["role"] == "agent"]) == 2


def test_reaction_corrects_a_wrong_readback_once():
    llm = FakeLLM({"agent": lambda i, t: fake_text("So your birthday is April 3, 1990?"),
                   "simulator": lambda i, t: fake_text("Yes. [DONE]")})
    sc = scenario(reactions=[{"when": r"april\s*3", "say": "No, March 4th."}])
    r = run_scenario(sc, [], llm, 0)
    said = [m["said"] for m in r.transcript if m["role"] == "patient"]
    assert said[1] == "No, March 4th." and said.count("No, March 4th.") == 1
