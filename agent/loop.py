"""Hand-rolled agent loop over the OpenAI Responses API (store=false; we own all state)."""

from __future__ import annotations

import json
from typing import Callable

from agent.ehr import MockEHR
from agent.prompt import build_system_prompt, build_tools
from agent.session import Session
from agent.tools import run_tool
from llm import config

FALLBACK_REPLY = "I'm sorry, I'm having trouble with that right now. Let me transfer you to our front desk."


class Agent:
    def __init__(self, ehr: MockEHR, llm, lessons: list | None = None, session: Session | None = None,
                 salt: str = "", on_tool: Callable[[str, dict, dict], None] | None = None):
        self.ehr, self.llm = ehr, llm
        self.session = session or Session()
        self.instructions = build_system_prompt(ehr, lessons or [])
        self.tools = build_tools(lessons or [])
        self.history: list[dict] = []
        self.trace: list[dict] = []
        self.turn = 0
        self.salt = salt
        self.on_tool = on_tool
        self.max_iters = config()["agent"]["max_tool_iterations"]

    def step(self, user_text: str) -> str:
        self.turn += 1
        self.session.last_user_message = user_text
        self.history.append({"role": "user", "content": user_text})
        for i in range(self.max_iters):
            res = self.llm.respond("agent", self.history, instructions=self.instructions,
                                   tools=self.tools, salt=f"{self.salt}:t{self.turn}:i{i}")
            self.history.extend(res.output)
            calls = res.function_calls
            if not calls:
                reply = res.text.strip() or FALLBACK_REPLY
                return reply
            for call in calls:
                try:
                    args = json.loads(call.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = run_tool(call["name"], args, self.session, self.ehr)
                self.trace.append({"turn": self.turn, "tool": call["name"], "args": args,
                                   "result": result, "session": self.session.to_dict()})
                if self.on_tool:
                    self.on_tool(call["name"], args, result)
                self.history.append({"type": "function_call_output", "call_id": call["call_id"],
                                     "output": json.dumps(result, default=str)})
        self.trace.append({"turn": self.turn, "tool": None, "event": "max_tool_iterations"})
        return FALLBACK_REPLY
