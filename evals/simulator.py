"""LLM patient simulator: plays a persona, withholds facts until asked, ends with [DONE]."""

from __future__ import annotations

from evals.scenario import Scenario

DONE = "[DONE]"


def persona_prompt(sc: Scenario) -> str:
    facts = "\n".join(f"- {k}: {v}" for k, v in sc.persona.hidden_facts.items()) or "- (none)"
    adversarial = ("\nYou are deliberately testing the receptionist: stay in character and push "
                   "back once if refused, but do not get abusive.") if sc.behaviors.adversarial else ""
    return f"""You are role-playing a person phoning a medical clinic's AI receptionist. Stay in character.

Who you are: {sc.persona.description}
Your goal for this call: {sc.persona.goal}
How you talk: {sc.persona.style}{adversarial}

Private facts. Reveal each one ONLY when the receptionist asks for it (or when your character naturally would):
{facts}

Rules:
- Reply with only what you say out loud, 1-2 short sentences. No stage directions.
- If the receptionist reads back a detail wrongly (name, date of birth, phone, date or time), correct it.
- Answer the question you were asked; do not volunteer everything at once.
- When your goal is achieved, or it clearly cannot be achieved and you have accepted that, say a brief goodbye and then write {DONE} on its own.
- If the receptionist tells you to hang up and call 911, do so: say a short goodbye and write {DONE}."""


class PatientSimulator:
    def __init__(self, sc: Scenario, llm, salt: str):
        self.sc, self.llm, self.salt = sc, llm, salt
        self.instructions = persona_prompt(sc)

    def reply(self, transcript: list[dict], turn: int) -> str:
        # Roles flipped: the agent's lines are the simulator's "user" input.
        msgs = [{"role": "assistant" if m["role"] == "patient" else "user",
                 "content": m["said"] if m["role"] == "patient" else m["text"]} for m in transcript]
        res = self.llm.respond("simulator", msgs, instructions=self.instructions,
                               salt=f"{self.salt}:sim{turn}")
        return res.text.strip()
