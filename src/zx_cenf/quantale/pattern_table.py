"""Pool per-pattern rule statistics into a lookup
table, learned ONCE from training diagrams only."""

from __future__ import annotations

from collections import defaultdict
from functools import cmp_to_key

from zx_cenf.quantale.pattern_hash import is_valid_pattern, make_context_key
from zx_cenf.quantale.strategies import _lexicographic_better


def build_pattern_rule_table(
    log_rows: list,
    min_occurrences: int = 5,
    tie_tolerance: float = 1e-9,
    require_competition: bool = True,
    min_distinct_sources: int | None = None,
) -> tuple:
    """
    Learn one preferred rewrite rule per recurring local pattern.

    The circuit valuation mu is a COST vector:

        (twoqubitcount, tcount, depth)

    Lower values are better.

    Therefore the stored delta is defined as:

        improvement = mu_before - mu_after

    Positive coordinates represent reductions in cost.
    """

    pooled: dict = defaultdict(list)
    pooled_sources: dict = defaultdict(set)

    for row in log_rows:
        # Only learn from actual local crossroads.
        if not row.was_crossroad:
            continue

        # Extraction must have succeeded so that mu_after is meaningful.
        if not row.mu_valid:
            continue

        if not is_valid_pattern(row.pattern_hash):
            continue

        if row.mu_before is None or row.mu_after is None:
            continue

        # IMPORTANT:
        # mu is a cost, so improvement is BEFORE - AFTER.
        #
        # Example:
        #   before = (10, 5, 20)
        #   after  = (8,  5, 18)
        #
        #   improvement = (2, 0, 2)
        #
        # Positive = better.
        improvement = tuple(
            a - b
            for a, b in zip(
                row.mu_before,
                row.mu_after,
            )
        )

        context_key = make_context_key(
            row.pattern_hash,
            row.competing_rules,
        )
        pooled[(context_key, row.rule)].append(improvement)
        if row.source_id is not None:
            pooled_sources[(context_key, row.rule)].add(row.source_id)

    by_pattern: dict = defaultdict(dict)
    counts: dict = defaultdict(dict)

    for (
        context_key,
        rule,
    ), improvements in pooled.items():

        has_enough_sources = (
            min_distinct_sources is None
            or len(pooled_sources[(context_key, rule)]) >= min_distinct_sources
        )
        if len(improvements) < min_occurrences or not has_enough_sources:
            continue

        n = len(improvements)

        avg = tuple(
            sum(
                improvement[i]
                for improvement in improvements
            ) / n
            for i in range(len(improvements[0]))
        )

        by_pattern[context_key][rule] = avg
        counts[context_key][rule] = n

    table: dict = {}

    diagnostics = {
        "n_patterns_pooled": len(
            {
                context_key
                for context_key, _rule
                in pooled.keys()
            }
        ),
        "n_patterns_with_enough_data": 0,
        "n_patterns_one_sided": 0,
        "n_patterns_incomplete": 0,
        "n_patterns_learned": 0,
        "n_patterns_dropped_as_tie": 0,
        "n_patterns_adequately_supported": 0,
        "n_patterns_insufficient_evidence": 0,
        "min_occurrences": min_occurrences,
        "min_distinct_sources": min_distinct_sources,
        "per_pattern_avg_deltas": dict(by_pattern),
        "per_pattern_counts": dict(counts),
        "per_pattern_observation_counts": {},
        "per_pattern_source_counts": {},
        "per_pattern_status": {},
    }

    all_context_keys = {context_key for context_key, _rule in pooled}
    for context_key in all_context_keys:
        diagnostics["per_pattern_observation_counts"][context_key] = {
            rule: len(pooled.get((context_key, rule), ()))
            for rule in context_key[1]
        }
        diagnostics["per_pattern_source_counts"][context_key] = {
            rule: len(pooled_sources.get((context_key, rule), ()))
            for rule in context_key[1]
        }
        diagnostics["per_pattern_status"][context_key] = "insufficient_evidence"

    for context_key, rule_improvements in by_pattern.items():
        diagnostics[
            "n_patterns_with_enough_data"
        ] += 1

        expected_rules = set(context_key[1])
        supported_rules = set(rule_improvements)
        is_one_sided = len(rule_improvements) < 2
        if is_one_sided:
            diagnostics["n_patterns_one_sided"] += 1

        if supported_rules != expected_rules:
            diagnostics["n_patterns_incomplete"] += 1
            if require_competition:
                continue

        # Only one rule has enough observations for this pattern.
        if is_one_sided:
            if require_competition:
                continue

            table[context_key] = next(
                iter(rule_improvements)
            )

            diagnostics[
                "n_patterns_learned"
            ] += 1

            continue

        diagnostics["n_patterns_adequately_supported"] += 1

        ranked = sorted(
            rule_improvements.items(),
            key=cmp_to_key(
                lambda a, b: _lexicographic_better(
                    a[1],
                    b[1],
                    tie_tolerance,
                )
            ),
            reverse=True,
        )

        best_rule, best_improvement = ranked[0]

        _runner_rule, runner_improvement = ranked[1]

        if (
            _lexicographic_better(
                best_improvement,
                runner_improvement,
                tie_tolerance,
            )
            > 0
        ):
            table[context_key] = best_rule

            diagnostics[
                "n_patterns_learned"
            ] += 1
            diagnostics["per_pattern_status"][context_key] = "learned"

        else:
            diagnostics[
                "n_patterns_dropped_as_tie"
            ] += 1
            diagnostics["per_pattern_status"][context_key] = "tie"

    diagnostics["n_patterns_insufficient_evidence"] = (
        diagnostics["n_patterns_pooled"]
        - diagnostics["n_patterns_adequately_supported"]
    )

    return table, diagnostics


def pattern_overlap_report(
    train_rows: list,
    test_context_keys,
    learned_table: dict | None = None,
) -> dict:
    train_keys = {
        make_context_key(row.pattern_hash, row.competing_rules)
        for row in train_rows
        if row.was_crossroad and is_valid_pattern(row.pattern_hash)
    }

    test_keys = {
        key
        for key in test_context_keys
        if isinstance(key, tuple) and is_valid_pattern(key[0])
    }
    train_hashes = {key[0] for key in train_keys}
    test_hashes = {key[0] for key in test_keys}
    learned_keys = set((learned_table or {}).keys())

    return {
        "n_unique_train_hashes": len(train_hashes),
        "n_unique_train_contexts": len(train_keys),
        "n_unique_test_hashes": len(test_hashes),
        "n_unique_test_contexts": len(test_keys),
        "n_structural_overlap": len(train_hashes & test_hashes),
        "n_decision_context_overlap": len(train_keys & test_keys),
        "n_learned_context_overlap": len(learned_keys & test_keys),
    }
