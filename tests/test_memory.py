"""Lessons memory: persistence, merge/supersede, retire, and the anti-overfit lint."""

from evals.scenario import load_scenarios
from improve.diagnose import LessonProposal, lint
from memory.compile import playbook_text
from memory.store import Lesson, LessonSource, LessonStore


def lesson(id, rule, status="accepted", merged_from=()):
    return Lesson(id=id, source=LessonSource(scenario="T01", run_id="r"), violated=[], evidence="",
                  root_cause="", target="playbook", trigger="x", rule=rule, status=status,
                  merged_from=list(merged_from))


def test_store_roundtrip_and_ids(tmp_path):
    s = LessonStore(tmp_path / "l.json")
    s.add(lesson(s.next_id(), "a"))
    s.add(lesson(s.next_id(), "b"))
    s.save()
    s2 = LessonStore(tmp_path / "l.json")
    assert [l.id for l in s2.lessons] == ["L-001", "L-002"] and s2.next_id() == "L-003"


def test_merged_lesson_supersedes_and_retired_lessons_drop_out(tmp_path):
    s = LessonStore(tmp_path / "l.json")
    s.add(lesson("L-001", "old rule"))
    s.add(lesson("L-002", "better rule", merged_from=["L-001"]))
    assert "old rule" not in playbook_text(s.accepted())
    s.set_status("L-002", "retired")
    assert "better rule" not in playbook_text(s.accepted())


def P(rule, target="playbook", tool=None):
    return LessonProposal(action="new", merge_into=None, target=target, tool=tool, trigger="a caller asks",
                          rule=rule, root_cause="", evidence="")


def test_lint_rejects_overfit_rules():
    scs = load_scenarios()
    assert lint(P("Always read the slot date back before confirming."), scs) == []
    assert lint(P("Verify James Whitfield before booking."), scs)
    assert lint(P("Offer October 16 when asked about next Friday."), scs)
    assert lint(P("x", target="tool_desc", tool="nope"), scs)
