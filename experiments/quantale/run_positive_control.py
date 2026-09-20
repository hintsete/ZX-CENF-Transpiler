"""Run the Phase F synthetic positive-control mechanism test."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from zx_cenf.quantale.evaluation import (
    _mu_lexicographically_better,
    evaluate_on_held_out_diagram,
)
from zx_cenf.quantale.pattern_table import build_pattern_rule_table
from zx_cenf.quantale.positive_control import (
    EXPECTED_PREFERRED_RULE,
    INTENTIONALLY_WRONG_RULE,
    TARGET_CONTEXT_KEY,
    TARGET_PATTERN_HASH,
    build_positive_control_graph,
    collect_target_action_observations,
)
from zx_cenf.quantale.strategies import (
    finalize_for_extraction,
    table_guided_reduce,
)
from zx_cenf.quantale.valuation import mu_from_pyzx_graph


OUT_DIR = Path("data/track3_results")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-size", type=int, default=20)
    parser.add_argument("--test-size", type=int, default=10)
    parser.add_argument("--min-occurrences", type=int, default=10)
    parser.add_argument("--oracle-orderings", type=int, default=5)
    return parser.parse_args()


def _terminal_mu_for_table(graph, table):
    work = graph.clone()
    outcome, trace = table_guided_reduce(work, table)
    finalize_for_extraction(work)
    return mu_from_pyzx_graph(work).coords, outcome, trace


def main() -> None:
    args = _parse_args()
    if args.train_size < args.min_occurrences:
        raise ValueError("train-size must be at least min-occurrences")
    if args.test_size < 1:
        raise ValueError("test-size must be positive")

    training_rows = []
    for index in range(args.train_size):
        graph = build_positive_control_graph(index, "train")
        rows = collect_target_action_observations(graph)
        training_rows.extend(rows)

    table, diagnostics = build_pattern_rule_table(
        training_rows,
        min_occurrences=args.min_occurrences,
    )
    learned_rule = table.get(TARGET_CONTEXT_KEY)

    print("Phase F positive control")
    print(
        f"  training embeddings: {args.train_size} | "
        f"target observations/rule: {args.train_size}"
    )
    print(
        f"  target context: {TARGET_PATTERN_HASH} | "
        f"rules={TARGET_CONTEXT_KEY[1]}"
    )
    print(
        f"  learned target rule: {learned_rule} | "
        f"expected: {EXPECTED_PREFERRED_RULE}"
    )
    print(
        "  average terminal improvements: "
        f"{diagnostics['per_pattern_avg_deltas'].get(TARGET_CONTEXT_KEY)}"
    )

    if learned_rule != EXPECTED_PREFERRED_RULE:
        raise RuntimeError(
            "POSITIVE CONTROL FAILED: the expected rule was not learned"
        )

    wrong_table = dict(table)
    wrong_table[TARGET_CONTEXT_KEY] = INTENTIONALLY_WRONG_RULE

    rows = []
    n_target_hits = 0
    n_interventions = 0
    n_lexicographic_wins = 0
    n_wrong_control_matches_greedy = 0

    for index in range(args.test_size):
        graph = build_positive_control_graph(index, "test")
        result = evaluate_on_held_out_diagram(
            graph,
            f"positive_control_test_{index:03d}",
            table,
            n_oracle_orderings=args.oracle_orderings,
            diagnostic=False,
        )
        wrong_mu, wrong_outcome, wrong_trace = _terminal_mu_for_table(
            graph,
            wrong_table,
        )

        target_trace = [
            record
            for record in result.table_trace
            if record.pattern_hash == TARGET_PATTERN_HASH
            and record.competing_rules == TARGET_CONTEXT_KEY[1]
        ]
        target_hits = len(target_trace)
        target_interventions = sum(
            1 for record in target_trace if record.differs_from_greedy_choice
        )
        learned_wins = (
            result.table_guided_mu is not None
            and result.naive_greedy_mu is not None
            and _mu_lexicographically_better(
                result.table_guided_mu,
                result.naive_greedy_mu,
            )
            > 0
        )
        wrong_matches_greedy = wrong_mu == result.naive_greedy_mu

        n_target_hits += target_hits
        n_interventions += target_interventions
        n_lexicographic_wins += int(learned_wins)
        n_wrong_control_matches_greedy += int(wrong_matches_greedy)

        if result.naive_greedy_mu is None or result.table_guided_mu is None:
            raise RuntimeError(
                "POSITIVE CONTROL FAILED: a held-out terminal valuation is missing"
            )

        delta = tuple(
            greedy - learned
            for greedy, learned in zip(
                result.naive_greedy_mu,
                result.table_guided_mu,
            )
        )
        print(
            f"  test {index:02d}: hits={target_hits}, "
            f"interventions={target_interventions}, "
            f"greedy={result.naive_greedy_mu}, "
            f"learned={result.table_guided_mu}, delta={delta}, "
            f"wrong={wrong_mu}"
        )

        rows.append(
            {
                "diagram_id": result.diagram_id,
                "target_hits": target_hits,
                "target_interventions": target_interventions,
                "greedy_mu": result.naive_greedy_mu,
                "learned_mu": result.table_guided_mu,
                "wrong_rule_mu": wrong_mu,
                "empirical_oracle_mu": result.oracle_mu,
                "full_reduce_mu": result.full_reduce_mu,
                "learned_stop": result.table_guided_stop,
                "wrong_stop": wrong_outcome.stop_reason.value,
                "wrong_hits": wrong_outcome.n_table_hits,
                "wrong_trace_size": len(wrong_trace),
                "lexicographic_win": learned_wins,
                "wrong_matches_greedy": wrong_matches_greedy,
            }
        )

    passed = (
        n_target_hits == args.test_size
        and n_interventions == args.test_size
        and n_lexicographic_wins == args.test_size
        and n_wrong_control_matches_greedy == args.test_size
    )

    print(
        "\n  summary: "
        f"target hits={n_target_hits}/{args.test_size}, "
        f"greedy interventions={n_interventions}/{args.test_size}, "
        f"lexicographic wins={n_lexicographic_wins}/{args.test_size}, "
        "wrong-rule control matched greedy="
        f"{n_wrong_control_matches_greedy}/{args.test_size}"
    )
    print(f"  verdict: {'PASS' if passed else 'FAIL'}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUT_DIR / "positive_control_summary.csv"
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote: {output_path}")

    if not passed:
        raise RuntimeError("POSITIVE CONTROL FAILED: held-out mechanism criteria failed")


if __name__ == "__main__":
    main()