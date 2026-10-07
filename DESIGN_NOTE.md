# Design note

**Agent.** A voice-ready receptionist with 10 scoped tools behind an `EHRAdapter`. **Safety is enforced in the tools, not the prompt:**
- the patient id comes from the verified session;
- every write needs verification plus the patient's own consent words, quoted verbatim;
- unauthorized proxies can't see appointments;
- duplicate registrations are refused;
- `escalate_to_human(urgent)` locks all writes.

**Evals.** 19 scenarios (13 train, 6 holdout). Score = 50% final records + 30% tool trace + 20% LLM judge. Safety hard gates are **decided by code**. A transcript-only judge can't see the database and proved inconsistent, so it only scores tone.

**Loop and memory.**
1. Failures are clustered.
2. The optimizer writes one structured lesson (trigger, rule, example) to `memory/lessons.json`.
3. A gate reruns all scenarios with it and accepts it only if the target improves, nothing that passed regresses, safety failures don't increase, and the holdout holds.

Lessons are reversible. Failures a prompt can't fix (e.g. formatting for speech) **go to a human as code fixes**.

**Result (demo set):**
- Accepted: "mention the late fee before cancelling within 24h". **Train pass rate 52% → 76%**; holdout flat.
- The gate **rejected** a lesson that raised missed emergencies (3 → 8).

**For a real clinic:** a human signs off on every lesson, and gates also run against de-identified real calls.

**AI vs. my judgment.**
- AI drafted most of the code.
- I chose the $7 budget with small models, an honest v0, taking safety verdicts away from the judge, and routing style failures to code.

More detail: [`AI_USAGE.md`](AI_USAGE.md).
