"""Deterministic speech-to-text noise. Free and reproducible (no LLM).

2care.ai agents hear patients through STT; digits and names are what it mangles most.
The agent sees the garbled text; the simulated patient knows what they really said and
corrects bad read-backs, which is exactly the behaviour we want to test.
"""

from __future__ import annotations

import random
import re

NUMBER_CONFUSIONS = {
    "thirteen": "thirty", "fourteen": "forty", "fifteen": "fifty", "sixteen": "sixty",
    "seventeen": "seventy", "eighteen": "eighty", "nineteen": "ninety",
    "13": "30", "14": "40", "15": "50", "16": "60", "17": "70", "18": "80", "19": "90",
    "13th": "30th", "14th": "40th", "15th": "50th", "16th": "60th",
}
FIRST_HIT_P, LATER_HIT_P = 0.75, 0.2


class ASRNoise:
    def __init__(self, seed: str, extra: dict[str, str] | None = None):
        self.rng = random.Random(seed)
        self.table = {**NUMBER_CONFUSIONS, **{k.lower(): v for k, v in (extra or {}).items()}}
        self.seen: set[str] = set()

    def __call__(self, text: str) -> str:
        def swap(m: re.Match) -> str:
            word = m.group(0)
            key = word.lower()
            if key not in self.table:
                return word
            p = LATER_HIT_P if key in self.seen else FIRST_HIT_P
            self.seen.add(key)
            if self.rng.random() >= p:
                return word
            out = self.table[key]
            return out.capitalize() if word[0].isupper() else out

        return re.sub(r"[A-Za-z0-9']+", swap, text)
