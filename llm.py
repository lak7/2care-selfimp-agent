"""Single gateway for every OpenAI call: cost accounting, global budget ledger, dev cache.

Nothing else in the repo imports `openai` directly. This is what keeps the whole
project inside a hard $7 budget.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import yaml

ROOT = Path(__file__).resolve().parent
LEDGER = ROOT / "runs" / "ledger.jsonl"
CACHE_DIR = ROOT / ".cache" / "llm"
_lock = threading.Lock()


@lru_cache
def config() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text())


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class LLMResult:
    output: list[dict]                 # raw output items (reasoning, function_call, message)
    text: str
    parsed: Any = None
    usage: dict = field(default_factory=dict)
    cost: float = 0.0
    cached: bool = False

    @property
    def function_calls(self) -> list[dict]:
        return [o for o in self.output if o.get("type") == "function_call"]


def ledger_total() -> float:
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(line)["usd"] for line in LEDGER.read_text().splitlines() if line.strip())


def price(model: str, usage: dict) -> float:
    p = config()["prices"][model]
    cached = usage.get("cached_input", 0)
    return ((usage.get("input", 0) - cached) * p["input"] + cached * p["cached_input"]
            + usage.get("output", 0) * p["output"]) / 1_000_000


class LLM:
    """One instance per run. Tracks run spend and enforces run + global caps."""

    def __init__(self, run_id: str = "adhoc", run_cap: float | None = None, use_cache: bool = True):
        b = config()["budget"]
        self.run_id = run_id
        self.run_cap = run_cap if run_cap is not None else b["default_run_cap_usd"]
        self.global_limit = b["global_cap_usd"] - b["safety_margin_usd"]
        self.use_cache = use_cache
        self.spent = 0.0
        self.by_role: dict[str, float] = {}
        self._client = None
        self._local = threading.local()

    @property
    def client(self):
        if self._client is None:
            from dotenv import load_dotenv
            from openai import OpenAI
            load_dotenv(ROOT / ".env")
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY missing: copy .env.example to .env")
            self._client = OpenAI(max_retries=6)  # backs off on 429s when running many conversations
        return self._client

    # Per-thread spend, so concurrent conversations each know their own cost.
    def start_tracking(self) -> None:
        self._local.spent = 0.0

    def tracked(self) -> float:
        return getattr(self._local, "spent", 0.0)

    def _check_budget(self) -> None:
        if self.spent >= self.run_cap:
            raise BudgetExceeded(f"run cap ${self.run_cap:.2f} reached (spent ${self.spent:.3f})")
        total = ledger_total()
        if total >= self.global_limit:
            raise BudgetExceeded(f"global ledger ${total:.3f} >= limit ${self.global_limit:.2f}")

    def respond(self, role: str, input: list[dict], instructions: str | None = None,
                tools: list[dict] | None = None, text_format: type | None = None,
                salt: str = "") -> LLMResult:
        cfg = config()["models"][role]
        params = {"model": cfg["model"], "input": input, "store": False,
                  "reasoning": {"effort": cfg["reasoning_effort"]},
                  "max_output_tokens": cfg["max_output_tokens"],
                  "include": ["reasoning.encrypted_content"]}
        if instructions:
            params["instructions"] = instructions
        if tools:
            params["tools"] = tools
            params["parallel_tool_calls"] = False

        key = hashlib.sha256(json.dumps(
            {**params, "schema": text_format.__name__ if text_format else None, "salt": salt},
            sort_keys=True, default=str).encode()).hexdigest()
        cache_file = CACHE_DIR / f"{key}.json"
        if self.use_cache and cache_file.exists():
            d = json.loads(cache_file.read_text())
            parsed = text_format.model_validate(d["parsed"]) if text_format and d["parsed"] else None
            return LLMResult(d["output"], d["text"], parsed, d["usage"], 0.0, cached=True)

        self._check_budget()
        if text_format:
            resp = self.client.responses.parse(text_format=text_format, **params)
            parsed = resp.output_parsed
        else:
            resp = self.client.responses.create(**params)
            parsed = None
        u = resp.usage
        usage = {"input": u.input_tokens, "output": u.output_tokens,
                 "cached_input": getattr(u.input_tokens_details, "cached_tokens", 0) or 0,
                 "reasoning": getattr(u.output_tokens_details, "reasoning_tokens", 0) or 0}
        cost = price(cfg["model"], usage)
        output = [o.model_dump(exclude_none=True, mode="json") for o in resp.output]
        result = LLMResult(output, resp.output_text or "", parsed, usage, cost)

        self._local.spent = getattr(self._local, "spent", 0.0) + cost
        with _lock:
            self.spent += cost
            self.by_role[role] = self.by_role.get(role, 0.0) + cost
            LEDGER.parent.mkdir(exist_ok=True)
            with LEDGER.open("a") as f:
                f.write(json.dumps({"run_id": self.run_id, "role": role, "model": cfg["model"],
                                    **usage, "usd": round(cost, 6)}) + "\n")
            if self.use_cache:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps({
                    "output": output, "text": result.text, "usage": usage,
                    "parsed": parsed.model_dump(mode="json") if parsed is not None else None}))
        return result


class FakeLLM:
    """Scripted stand-in for tests: `script[role]` is a callable(input, tools) -> LLMResult."""

    def __init__(self, script: dict[str, Callable[..., LLMResult]]):
        self.script = script
        self.spent = 0.0
        self.by_role: dict[str, float] = {}
        self.calls: list[tuple[str, list]] = []
        self._lock = threading.Lock()

    def start_tracking(self) -> None:
        pass

    def tracked(self) -> float:
        return 0.0

    def respond(self, role, input, instructions=None, tools=None, text_format=None, salt=""):
        with self._lock:
            self.calls.append((role, list(input)))
            return self.script[role](input, tools)


def fake_text(text: str) -> LLMResult:
    return LLMResult([{"type": "message", "role": "assistant",
                       "content": [{"type": "output_text", "text": text}]}], text)


def fake_call(name: str, args: dict, call_id: str) -> LLMResult:
    return LLMResult([{"type": "function_call", "name": name, "arguments": json.dumps(args),
                       "call_id": call_id}], "")
