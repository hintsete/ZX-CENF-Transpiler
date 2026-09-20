"""orchestration -- LEAVE-ONE-OUT over the real diagram population."""

from pathlib import Path

import pyzx as zx

from zx_cenf.quantale.evaluation import evaluate_on_held_out_diagram
from zx_cenf.quantale.instrumented_reduction import (
    DEFAULT_CONTEXT_REPRESENTATION,
    run_instrumented_reduction,
)
from zx_cenf.quantale.pattern_table import (
    build_pattern_rule_table,
    pattern_overlap_report,
)


DATA_DIR = Path("data/toy")

DIAGRAM_NAMES = [
    "toy_larger_clifford_heavy",
    "toy_medium_diverse",
    "toy_small_dense",
]

N_TRAINING_SWEEPS_PER_DIAGRAM = 5
MIN_OCCURRENCES_PER_RULE = 5


def train_pattern_table(train_names):
    all_log_rows = []

    for name in train_names:
        circuit = zx.Circuit.load(
            str(DATA_DIR / f"{name}.qasm")
        )

        for sweep_seed in range(
            N_TRAINING_SWEEPS_PER_DIAGRAM
        ):
            g = circuit.to_graph()

            log_rows, outcome = run_instrumented_reduction(
                g,
                seed=sweep_seed,
            )

            print(
                f"    [{name}] sweep {sweep_seed}: "
                f"{outcome.n_steps} applied rewrites, "
                f"{len(log_rows)} logged action observations, "
                f"stop={outcome.stop_reason.value}"
            )

            all_log_rows.extend(log_rows)

    table, diagnostics = build_pattern_rule_table(
        all_log_rows,
        min_occurrences=MIN_OCCURRENCES_PER_RULE,
    )

    return (
        table,
        diagnostics,
        all_log_rows,
    )


def main():
    print(
        "Phase F configuration: "
        f"context={DEFAULT_CONTEXT_REPRESENTATION}, "
        "reward=terminal μ delta after forced rule + greedy continuation, "
        "priority=(two-qubit count, depth, T-count)"
    )

    for held_out in DIAGRAM_NAMES:
        train_names = [
            name
            for name in DIAGRAM_NAMES
            if name != held_out
        ]

        print(f"\n{'=' * 72}")
        print(
            f"HELD OUT: {held_out}   "
            f"(trained on: {train_names})"
        )
        print(f"{'=' * 72}")

        table, diagnostics, all_log_rows = (
            train_pattern_table(train_names)
        )

        n_crossroad_observations = sum(
            1
            for row in all_log_rows
            if row.was_crossroad
        )

        n_valid = sum(
            1
            for row in all_log_rows
            if row.was_crossroad and row.mu_valid
        )

        pct_valid = (
            100 * n_valid / n_crossroad_observations
            if n_crossroad_observations
            else 0.0
        )

        print(
            f"  Logged {n_crossroad_observations} counterfactual "
            f"crossroad action observations | terminal μ valid: "
            f"{n_valid} ({pct_valid:.1f}%)"
        )

        print(
            f"  Patterns pooled: "
            f"{diagnostics['n_patterns_pooled']} | "
            f"enough data: "
            f"{diagnostics['n_patterns_with_enough_data']} | "
            f"incomplete action support: "
            f"{diagnostics['n_patterns_incomplete']} | "
            f"ties: "
            f"{diagnostics['n_patterns_dropped_as_tie']}"
        )

        print(
            "  EVIDENCE-BACKED learned preferences: "
            f"{len(table)}"
        )

        circuit = zx.Circuit.load(
            str(DATA_DIR / f"{held_out}.qasm")
        )

        g = circuit.to_graph()

        result = evaluate_on_held_out_diagram(
            g,
            held_out,
            table,
        )

        overlap = pattern_overlap_report(
            all_log_rows,
            result.test_context_keys,
            learned_table=table,
        )

        print(
            "\n  --- Pattern overlap diagnostic ---"
        )

        print(
            "  unique train hashes: "
            f"{overlap['n_unique_train_hashes']} "
            f"({overlap['n_unique_train_contexts']} decision contexts)"
        )

        print(
            "  unique test hashes:  "
            f"{overlap['n_unique_test_hashes']} "
            f"({overlap['n_unique_test_contexts']} decision contexts)"
        )

        print(
            "  structural/context/learned overlap: "
            f"{overlap['n_structural_overlap']}/"
            f"{overlap['n_decision_context_overlap']}/"
            f"{overlap['n_learned_context_overlap']}"
        )

        total_crossroad_lookups = (
            result.table_hits
            + result.table_misses
        )

        hit_rate = (
            result.table_hits / total_crossroad_lookups
            if total_crossroad_lookups
            else 0.0
        )

        print(
            f"\n  --- Held-out evaluation: "
            f"{held_out} ---"
        )

        print(
            f"  empirical oracle: "
            f"{result.oracle_mu}"
        )

        print(
            f"  naive greedy:     "
            f"{result.naive_greedy_mu}  "
            f"[{result.naive_greedy_stop}]"
        )

        print(
            f"  table-guided:     "
            f"{result.table_guided_mu}  "
            f"[{result.table_guided_stop}]"
        )

        print(
            f"  pyzx full_reduce: "
            f"{result.full_reduce_mu}"
        )

        print(
            "  table crossroad coverage: "
            f"{result.table_hits}/"
            f"{total_crossroad_lookups} = "
            f"{hit_rate:.1%}"
        )

        if not table:
            print(
                "  coverage interpretation: no supported non-tied "
                "preference was learned, so no table action was available"
            )

        print(
            "  learned choices differing "
            "from greedy: "
            f"{result.table_n_differs_from_greedy}"
        )


if __name__ == "__main__":
    main()