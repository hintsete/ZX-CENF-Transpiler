"""Pool per-pattern rule statistics into a lookup
table, learned ONCE from training diagrams only."""

from __future__ import annotations

from collections import defaultdict
from functools import cmp_to_key

from zx_cenf.quantale.pattern_hash import is_valid_pattern
from zx_cenf.quantale.strategies import _lexicographic_better


def build_pattern_rule_table(
    log_rows: list,
    min_occurrences: int = 5,
    tie_tolerance: float = 1e-9,
    require_competition: bool = True,
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

        pooled[
            (
                row.pattern_hash,
                row.rule,
            )
        ].append(improvement)

    by_pattern: dict = defaultdict(dict)
    counts: dict = defaultdict(dict)

    for (
        pattern_hash,
        rule,
    ), improvements in pooled.items():

        if len(improvements) < min_occurrences:
            continue

        n = len(improvements)

        avg = tuple(
            sum(
                improvement[i]
                for improvement in improvements
            ) / n
            for i in range(len(improvements[0]))
        )

        by_pattern[pattern_hash][rule] = avg
        counts[pattern_hash][rule] = n

    table: dict = {}

    diagnostics = {
        "n_patterns_pooled": len(
            {
                pattern_hash
                for pattern_hash, _rule
                in pooled.keys()
            }
        ),
        "n_patterns_with_enough_data": 0,
        "n_patterns_one_sided": 0,
        "n_patterns_learned": 0,
        "n_patterns_dropped_as_tie": 0,
        "per_pattern_avg_deltas": dict(by_pattern),
        "per_pattern_counts": dict(counts),
    }

    for pattern_hash, rule_improvements in by_pattern.items():
        diagnostics[
            "n_patterns_with_enough_data"
        ] += 1

        # Only one rule has enough observations for this pattern.
        if len(rule_improvements) < 2:
            diagnostics[
                "n_patterns_one_sided"
            ] += 1

            if require_competition:
                continue

            table[pattern_hash] = next(
                iter(rule_improvements)
            )

            diagnostics[
                "n_patterns_learned"
            ] += 1

            continue

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
            table[pattern_hash] = best_rule

            diagnostics[
                "n_patterns_learned"
            ] += 1

        else:
            diagnostics[
                "n_patterns_dropped_as_tie"
            ] += 1

    return table, diagnostics


def pattern_overlap_report(
    train_rows: list,
    test_hashes: list,
) -> dict:
    train_hashes = {
        row.pattern_hash
        for row in train_rows
        if is_valid_pattern(row.pattern_hash)
    }

    train_crossroad_hashes = {
        row.pattern_hash
        for row in train_rows
        if (
            row.was_crossroad
            and is_valid_pattern(row.pattern_hash)
        )
    }

    test_set = {
        pattern_hash
        for pattern_hash in test_hashes
        if is_valid_pattern(pattern_hash)
    }

    return {
        "n_unique_train_hashes": len(train_hashes),
        "n_unique_train_crossroad_hashes": len(
            train_crossroad_hashes
        ),
        "n_unique_test_hashes": len(test_set),
        "n_overlap_all": len(
            train_hashes & test_set
        ),
        "n_overlap_crossroad": len(
            train_crossroad_hashes & test_set
        ),
    }