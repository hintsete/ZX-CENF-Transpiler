"""Null/randomized pattern table."""

from __future__ import annotations

import random


def build_null_table(learned_table: dict, diagnostics: dict, seed: int) -> dict:
    rng = random.Random(seed)
    per_pattern = diagnostics.get("per_pattern_avg_deltas", {})

    null_table = {}
    for pattern_hash in sorted(learned_table.keys()):
        candidate_rules = sorted(per_pattern.get(pattern_hash, {}).keys())
        if not candidate_rules:
            null_table[pattern_hash] = learned_table[pattern_hash]
            continue
        null_table[pattern_hash] = rng.choice(candidate_rules)

    return null_table