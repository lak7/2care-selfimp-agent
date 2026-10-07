# Where AI helped, and where our judgment overrode it

Logged during the build, not reconstructed at the end.

## AI helped
- **Brainstorming and PRD drafting:** Claude (Claude Code) proposed the overall architecture: code-enforced guardrails, a 3-layer eval, a lessons memory with a gate, and a train/holdout split. It drafted `PRD.md` from decisions made in a Q&A.
- **Scaffolding and most code:** the tools, checks, runner, report and loop were drafted with Claude Code and then reviewed and adjusted.
- **Research:** current OpenAI model pricing (official pricing page) and 2care.ai's product (website, Cliniko listing).

## Where our judgment overrode the AI
- **Budget and models.** The AI first suggested Claude models, then `gpt-5.6-terra` as judge, optimizer and simulator. We cut the budget to **$7 total** and moved to `gpt-5-nano` (agent, simulator) and `gpt-5-mini` (judge, optimizer).
  - We also rejected a Terra simulator, which would have spent about half the budget on the simulator alone.
  - The cost ledger, the hard cap and the dev cache follow from this decision.
- **Scope.** The AI recommended making new-patient registration a stretch goal (depth over breadth). We made it a core goal, because intake is central to what 2care.ai sells.
- **Company context.** We asked for research on 2care.ai, which reshaped the design:
  - replies that work when spoken (2care is voice-first);
  - an `EHRAdapter` boundary (they write back to about 95 EHR systems);
  - specialties matching their customers;
  - the DD/MM vs MM/DD date-of-birth ambiguity (they operate in the US, EU and India).
- **Baseline honesty.** We chose an honest minimal v0 prompt over a deliberately weakened one, so the before/after reflects real failures rather than staged ones.

## Decisions made during the build
- **Speech-to-text noise is simulated deterministically in code, not by an LLM.** It's free and reproducible, and the simulated patient still knows what they really said, so it corrects bad read-backs.
- **Phantom-booking and invented-time detection is deterministic.** Regexes over agent text are cross-checked against tool results, rather than left to the judge, because a transcript-only judge can't see the database.
- **The anti-overfit lint is in code.** Lessons naming patients, dates or scenario ids are rejected, instead of relying on the optimizer prompt alone.
