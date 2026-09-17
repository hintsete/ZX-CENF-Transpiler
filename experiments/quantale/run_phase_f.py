"""orchestration -- LEAVE-ONE-OUT over the real diagram population."""

from pathlib import Path

import pyzx as zx

from zx_cenf.quantale.evaluation import evaluate_on_held_out_diagram
from zx_cenf.quantale.instrumented_reduction import (
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

            log_rows, reached = run_instrumented_reduction(
                g,
                seed=sweep_seed,
            )

            print(
                f"    [{name}] sweep {sweep_seed}: "
                f"{len(log_rows)} steps, "
                f"fixed_point={reached}"
            )

            all_log_rows.extend(log_rows)

    table, diagnostics = build_pattern_rule_table(
        all_log_rows,
        min_occurrences=5,
    )

    return (
        table,
        diagnostics,
        all_log_rows,
    )


def main():
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

        n_crossroad = sum(
            1
            for row in all_log_rows
            if row.was_crossroad
        )

        n_valid = sum(
            1
            for row in all_log_rows
            if row.mu_valid
        )

        pct_crossroad = (
            100 * n_crossroad / len(all_log_rows)
            if all_log_rows
            else 0.0
        )

        print(
            f"  Logged {len(all_log_rows)} applications | "
            f"LOCAL crossroads: "
            f"{n_crossroad} "
            f"({pct_crossroad:.1f}%) | "
            f"mu_valid: {n_valid}"
        )

        print(
            f"  Patterns pooled: "
            f"{diagnostics['n_patterns_pooled']} | "
            f"enough data: "
            f"{diagnostics['n_patterns_with_enough_data']} | "
            f"one-sided (NOT learned): "
            f"{diagnostics['n_patterns_one_sided']} | "
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
            result.test_pattern_hashes,
        )

        print(
            "\n  --- Pattern overlap diagnostic ---"
        )

        print(
            "  unique train hashes: "
            f"{overlap['n_unique_train_hashes']} "
            "(crossroad-only: "
            f"{overlap['n_unique_train_crossroad_hashes']})"
        )

        print(
            "  unique test hashes:  "
            f"{overlap['n_unique_test_hashes']}"
        )

        print(
            "  OVERLAP: "
            f"{overlap['n_overlap_all']} "
            "(crossroad-only: "
            f"{overlap['n_overlap_crossroad']})"
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
            f"  oracle:           "
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

        print(
            "  learned choices differing "
            "from greedy: "
            f"{sum(1 for row in [])}"
        )


if __name__ == "__main__":
    main()