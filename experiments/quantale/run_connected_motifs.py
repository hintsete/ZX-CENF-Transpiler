"""Run the Phase F connected multi-motif population experiment."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from zx_cenf.quantale.connected_motifs import (
    MOTIF_SPECS,
    build_connected_motif_graph,
    collect_target_action_observations,
)
from zx_cenf.quantale.evaluation import (
    _mu_lexicographically_better,
    evaluate_on_held_out_diagram,
)
from zx_cenf.quantale.pattern_table import build_pattern_rule_table
from zx_cenf.quantale.strategies import finalize_for_extraction, table_guided_reduce
from zx_cenf.quantale.valuation import mu_from_pyzx_graph


OUT_DIR = Path("data/track3_results")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-per-family", type=int, default=12)
    parser.add_argument("--test-per-family", type=int, default=6)
    parser.add_argument("--min-occurrences", type=int, default=8)
    parser.add_argument("--oracle-orderings", type=int, default=5)
    return parser.parse_args()


def _winner(rows) -> str:
    valid = [row for row in rows if row.mu_valid and row.mu_after is not None]
    if not valid:
        return "invalid"
    ordered = sorted(valid, key=lambda row: (row.mu_after[0], row.mu_after[2], row.mu_after[1]))
    if len(ordered) > 1 and ordered[0].mu_after == ordered[1].mu_after:
        return "tie"
    return ordered[0].rule


def _terminal_mu_for_table(graph, table):
    work = graph.clone()
    outcome, trace = table_guided_reduce(work, table)
    if not finalize_for_extraction(work):
        return None, outcome, trace
    return mu_from_pyzx_graph(work).coords, outcome, trace


def _comparison(left, right) -> str:
    if left is None or right is None:
        return "invalid"
    result = _mu_lexicographically_better(left, right)
    return "win" if result > 0 else "loss" if result < 0 else "tie"


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    if args.train_per_family < args.min_occurrences:
        raise ValueError("train-per-family must be at least min-occurrences")
    if args.test_per_family < 1:
        raise ValueError("test-per-family must be positive")

    print("Phase F connected multi-motif experiment")
    print(
        f"  families={len(MOTIF_SPECS)}, "
        f"train/family={args.train_per_family}, "
        f"test/family={args.test_per_family}, "
        f"min occurrences/rule={args.min_occurrences}"
    )

    training_rows = []
    training_csv_rows = []
    winner_counts = defaultdict(Counter)

    for spec in MOTIF_SPECS:
        for index in range(args.train_per_family):
            graph = build_connected_motif_graph(spec, index, "train")
            rows = collect_target_action_observations(graph, spec)
            training_rows.extend(rows)
            winner = _winner(rows)
            winner_counts[spec.name][winner] += 1
            for row in rows:
                training_csv_rows.append(
                    {
                        "family": spec.name,
                        "embedding": index,
                        "pattern_hash": row.pattern_hash,
                        "competing_rules": row.competing_rules,
                        "rule": row.rule,
                        "mu_before": row.mu_before,
                        "mu_after": row.mu_after,
                        "mu_valid": row.mu_valid,
                        "embedding_winner": winner,
                    }
                )

    table, diagnostics = build_pattern_rule_table(
        training_rows,
        min_occurrences=args.min_occurrences,
    )

    print(
        "  training: "
        f"pooled={diagnostics['n_patterns_pooled']}, "
        f"enough_data={diagnostics['n_patterns_with_enough_data']}, "
        f"learned={diagnostics['n_patterns_learned']}, "
        f"ties={diagnostics['n_patterns_dropped_as_tie']}, "
        f"incomplete={diagnostics['n_patterns_incomplete']}"
    )
    for spec in MOTIF_SPECS:
        print(
            f"  {spec.name:24s} winners={dict(winner_counts[spec.name])} "
            f"learned={table.get(spec.context_key)} "
            f"expected={spec.expected_preferred_rule}"
        )

    result_rows = []
    aggregate = defaultdict(Counter)

    for spec in MOTIF_SPECS:
        wrong_table = dict(table)
        wrong_table[spec.context_key] = spec.intentionally_wrong_rule

        for index in range(args.test_per_family):
            graph = build_connected_motif_graph(spec, index, "test")
            result = evaluate_on_held_out_diagram(
                graph,
                f"{spec.name}_test_{index:03d}",
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
                if (
                    record.pattern_hash == spec.pattern_hash
                    and record.competing_rules == spec.context_key[1]
                )
            ]
            target_wrong_trace = [
                record
                for record in wrong_trace
                if (
                    record.pattern_hash == spec.pattern_hash
                    and record.competing_rules == spec.context_key[1]
                )
            ]
            learned_vs_greedy = _comparison(
                result.table_guided_mu,
                result.naive_greedy_mu,
            )
            learned_vs_wrong = _comparison(result.table_guided_mu, wrong_mu)
            target_hits = len(target_trace)
            interventions = sum(
                record.differs_from_greedy_choice for record in target_trace
            )

            aggregate[spec.name]["tests"] += 1
            aggregate[spec.name]["target_hits"] += int(target_hits > 0)
            aggregate[spec.name]["interventions"] += int(interventions > 0)
            aggregate[spec.name][f"greedy_{learned_vs_greedy}"] += 1
            aggregate[spec.name][f"wrong_{learned_vs_wrong}"] += 1

            result_rows.append(
                {
                    "family": spec.name,
                    "diagram_id": result.diagram_id,
                    "expected_rule": spec.expected_preferred_rule,
                    "learned_rule": table.get(spec.context_key),
                    "wrong_rule": spec.intentionally_wrong_rule,
                    "initial_context_present": True,
                    "target_hits": target_hits,
                    "target_interventions": interventions,
                    "wrong_target_hits": len(target_wrong_trace),
                    "greedy_mu": result.naive_greedy_mu,
                    "learned_mu": result.table_guided_mu,
                    "wrong_mu": wrong_mu,
                    "oracle_mu": result.oracle_mu,
                    "full_reduce_mu": result.full_reduce_mu,
                    "learned_vs_greedy": learned_vs_greedy,
                    "learned_vs_wrong": learned_vs_wrong,
                    "learned_stop": result.table_guided_stop,
                    "wrong_stop": wrong_outcome.stop_reason.value,
                    "all_table_hits": result.table_hits,
                    "all_table_misses": result.table_misses,
                    "wrong_all_hits": wrong_outcome.n_table_hits,
                }
            )

    family_rows = []
    totals = Counter()
    print("\n  held-out family results")
    for spec in MOTIF_SPECS:
        counts = aggregate[spec.name]
        totals.update(counts)
        tests = counts["tests"]
        row = {
            "family": spec.name,
            "training_winners": dict(winner_counts[spec.name]),
            "expected_rule": spec.expected_preferred_rule,
            "learned_rule": table.get(spec.context_key),
            "tests": tests,
            "initial_structural_recurrence": 1.0,
            "target_trajectory_coverage": counts["target_hits"] / tests,
            "intervention_rate": counts["interventions"] / tests,
            "vs_greedy_wins": counts["greedy_win"],
            "vs_greedy_ties": counts["greedy_tie"],
            "vs_greedy_losses": counts["greedy_loss"],
            "vs_wrong_wins": counts["wrong_win"],
            "vs_wrong_ties": counts["wrong_tie"],
            "vs_wrong_losses": counts["wrong_loss"],
        }
        family_rows.append(row)
        print(
            f"  {spec.name:24s} structural=100%, "
            "trajectory hit="
            f"{row['target_trajectory_coverage']:.0%}, "
            f"intervention={row['intervention_rate']:.0%}, "
            "vs greedy W/T/L="
            f"{row['vs_greedy_wins']}/{row['vs_greedy_ties']}/"
            f"{row['vs_greedy_losses']}, vs wrong W/T/L="
            f"{row['vs_wrong_wins']}/{row['vs_wrong_ties']}/"
            f"{row['vs_wrong_losses']}"
        )

    total_tests = totals["tests"]
    print(
        "\n  overall: "
        f"learned contexts={len(table)}/{len(MOTIF_SPECS)}, "
        "initial structural recurrence=100.0%, "
        f"target trajectory hits={totals['target_hits']}/{total_tests} "
        f"({totals['target_hits'] / total_tests:.1%}), "
        f"interventions={totals['interventions']}/{total_tests} "
        f"({totals['interventions'] / total_tests:.1%})"
    )
    print(
        "  learned vs greedy W/T/L="
        f"{totals['greedy_win']}/{totals['greedy_tie']}/"
        f"{totals['greedy_loss']} | learned vs wrong W/T/L="
        f"{totals['wrong_win']}/{totals['wrong_tie']}/"
        f"{totals['wrong_loss']}"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = (
        (OUT_DIR / "connected_motif_training.csv", training_csv_rows),
        (OUT_DIR / "connected_motif_results.csv", result_rows),
        (OUT_DIR / "connected_motif_family_summary.csv", family_rows),
    )
    for path, rows in paths:
        _write_csv(path, rows)
        print(f"  wrote: {path}")


if __name__ == "__main__":
    main()