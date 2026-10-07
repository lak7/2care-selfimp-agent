"""Whole pipeline (runner → checks → judge → diagnose → screen → report) on a scripted model, $0."""

import evals.report
import evals.runner
import improve.loop
from evals.judge import ItemScore, Verdict
from improve.diagnose import LessonProposal
from llm import FakeLLM, LLMResult, fake_text
from memory.store import LessonStore


def fake_llm():
    def judge(input, tools):
        v = Verdict(items=[ItemScore(id=i, score=2, evidence="ok") for i in
                           ("clarity", "one_question", "empathy", "minimal_phi")],
                    medical_advice=False, medical_advice_evidence="", phi_disclosed=False, phi_evidence="")
        return LLMResult([], "", parsed=v)

    def optimizer(input, tools):
        return LLMResult([], "", parsed=LessonProposal(
            action="new", merge_into=None, target="playbook", tool=None, trigger="a caller wants an appointment",
            rule="Verify identity first, then search and offer real slots.", example=None, root_cause="r", evidence="e"))

    sim_turns = {}

    def simulator(input, tools):
        n = len(input)
        return fake_text("Thanks, bye. [DONE]" if n >= 3 else "My name is Olivia Bennett.")

    return FakeLLM({"agent": lambda i, t: fake_text("Sure, I can help with that."),
                    "simulator": simulator, "judge": judge, "optimizer": optimizer})


def test_improve_loop_end_to_end_with_fake_model(tmp_path, monkeypatch):
    fake = fake_llm()
    for mod in (evals.runner, evals.report, improve.loop):
        monkeypatch.setattr(mod, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(improve.loop, "LLM", lambda **kw: fake)
    store_path = tmp_path / "lessons.json"
    monkeypatch.setattr(improve.loop, "LessonStore", lambda: LessonStore(store_path))

    report = improve.loop.improve(rounds=1, k=1, max_candidates=1)

    text = open(report).read()
    assert "Improvement loop" in text and "L-001" in text
    lessons = LessonStore(store_path).lessons
    assert lessons and lessons[0].status == "rejected"   # a do-nothing agent cannot improve → gate holds
    assert {r for r, _ in fake.calls} == {"agent", "simulator", "judge", "optimizer"}


def test_improve_reuses_latest_matching_eval_run(tmp_path, monkeypatch):
    from evals.runner import run_suite
    from evals.scenario import load_scenarios

    fake = fake_llm()
    for mod in (evals.runner, evals.report, improve.loop):
        monkeypatch.setattr(mod, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(improve.loop, "LLM", lambda **kw: fake)
    monkeypatch.setattr(improve.loop, "LessonStore", lambda: LessonStore(tmp_path / "lessons.json"))

    prior = run_suite(load_scenarios("all"), [], fake, 1, "all", "evals")      # what `evals run` saves
    run_suite(load_scenarios("train", ["T01"]), [], fake, 1, "train", "partial")  # subset: never a baseline
    improve.loop.improve(rounds=1, k=1, max_candidates=1)
    assert not [d for d in tmp_path.iterdir() if d.name.endswith("-baseline")]
    assert LessonStore(tmp_path / "lessons.json").lessons[0].source.run_id == prior.run_id

    improve.loop.improve(rounds=1, k=1, max_candidates=1, fresh_baseline=True)
    assert [d for d in tmp_path.iterdir() if d.name.endswith("-baseline")]
