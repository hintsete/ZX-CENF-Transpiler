"""Null/randomized pattern table."""

from __future__ import annotations

import random
from functools import cmp_to_key

from zx_cenf.quantale.strategies import _lexicographic_better


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


def build_wrong_rule_table(learned_table: dict, diagnostics: dict) -> dict:
    """Select the empirically worst supported alternative at learned keys."""
    per_pattern = diagnostics.get("per_pattern_avg_deltas", {})
    wrong_table = {}
    for context_key, learned_rule in learned_table.items():
        alternatives = [
            item
            for item in per_pattern.get(context_key, {}).items()
            if item[0] != learned_rule
        ]
        if not alternatives:
            continue
        alternatives.sort(
            key=cmp_to_key(lambda a, b: _lexicographic_better(a[1], b[1]))
        )
        wrong_table[context_key] = alternatives[0][0]
    return wrong_table
