"""Run the larger, non-planted Phase F natural-population experiment."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from zx_cenf.quantale.evaluation import (
    _mu_lexicographically_better,
    evaluate_ablation,
)
from zx_cenf.quantale.instrumented_reduction import run_instrumented_reduction
from zx_cenf.quantale.natural_population import (
    audit_train_test_separation,
    build_natural_circuit,
    circuit_fingerprint,
    make_population_specs,
)
from zx_cenf.quantale.null_table import build_null_table, build_wrong_rule_table
from zx_cenf.quantale.pattern_hash import make_context_key
from zx_cenf.quantale.pattern_table import build_pattern_rule_table
from zx_cenf.quantale.strategies import finalize_for_extraction, table_guided_reduce
from zx_cenf.quantale.valuation import mu_from_pyzx_graph


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-per-family", type=int, default=10)
    parser.add_argument("--test-per-family", type=int, default=5)
    parser.add_argument("--training-sweeps", type=int, default=1)
    parser.add_argument("--min-occurrences", type=int, default=5)
    parser.add_argument("--min-source-circuits", type=int, default=5)
    parser.add_argument("--oracle-orderings", type=int, default=10)
    parser.add_argument("--null-seed", type=int, default=42)
    parser.add_argument("--base-seed", type=int, default=17_000)
    parser.add_argument("--out-dir", type=Path, default=Path("data/track3_results/natural"))
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _comparison(left, right) -> str:
    if left is None or right is None:
        return "invalid"
    result = _mu_lexicographically_better(left, right)
    return "win" if result > 0 else "loss" if result < 0 else "tie"


def _terminal_for_table(graph, table):
    work = graph.clone()
    outcome, trace = table_guided_reduce(work, table)
    if not finalize_for_extraction(work):
        return None, outcome, trace
    try:
        return mu_from_pyzx_graph(work).coords, outcome, trace
    except Exception:
        return None, outcome, trace


def _context_label(context_key) -> str:
    return f"{context_key[0]}|{'|'.join(context_key[1])}"


def main() -> None:
    args = _parse_args()
    if args.min_occurrences < 5:
        raise ValueError("this experiment must not lower the existing five-observation threshold")
    if args.min_source_circuits < 2:
        raise ValueError("min-source-circuits must demonstrate cross-circuit recurrence")
    if min(args.train_per_family, args.test_per_family, args.training_sweeps) < 1:
        raise ValueError("population sizes and training sweeps must be positive")

    specs = make_population_specs(
        args.train_per_family,
        args.test_per_family,
        base_seed=args.base_seed,
    )
    circuits = [(spec, build_natural_circuit(spec)) for spec in specs]
    audit = audit_train_test_separation(circuits)
    if audit["n_exact_train_test_duplicates"] or audit["n_near_duplicate_train_test_pairs"]:
        raise RuntimeError(f"train/test separation audit failed: {audit}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for spec, circuit in circuits:
        manifest_rows.append(
            {
                **vars(spec),
                "gate_count": len(circuit.gates),
                "fingerprint": circuit_fingerprint(circuit),
            }
        )
    _write_csv(args.out_dir / "population_manifest.csv", manifest_rows)

    train = [(spec, circuit) for spec, circuit in circuits if spec.split == "train"]
    test = [(spec, circuit) for spec, circuit in circuits if spec.split == "test"]
    print("Phase F natural-population experiment")
    print(
        f"  circuits={len(circuits)} (train={len(train)}, test={len(test)}), "
        f"families={len({spec.family for spec in specs})}, sweeps={args.training_sweeps}"
    )
    print(
        "  separation audit: exact train/test duplicates=0, "
        "near duplicates=0, max gate-sequence similarity="
        f"{audit['max_train_test_similarity']:.3f}"
    )

    training_rows = []
    training_stops = Counter()
    training_run_rows = []
    for spec, circuit in train:
        for sweep in range(args.training_sweeps):
            rows, outcome = run_instrumented_reduction(
                circuit.to_graph(),
                seed=sweep,
                source_id=spec.circuit_id,
            )
            training_rows.extend(rows)
            training_stops[outcome.stop_reason.value] += 1
            crossroad_rows = [row for row in rows if row.was_crossroad]
            training_run_rows.append(
                {
                    "circuit_id": spec.circuit_id,
                    "family": spec.family,
                    "sweep": sweep,
                    "steps": outcome.n_steps,
                    "stop_reason": outcome.stop_reason.value,
                    "crossroad_action_observations": len(crossroad_rows),
                    "valid_observations": sum(row.mu_valid for row in crossroad_rows),
                }
            )
            print(
                f"  train {spec.circuit_id} sweep={sweep}: steps={outcome.n_steps}, "
                f"crossroad actions={len(crossroad_rows)}, stop={outcome.stop_reason.value}"
            )
    _write_csv(args.out_dir / "training_runs.csv", training_run_rows)

    table, diagnostics = build_pattern_rule_table(
        training_rows,
        min_occurrences=args.min_occurrences,
        min_distinct_sources=args.min_source_circuits,
    )
    null_table = build_null_table(table, diagnostics, seed=args.null_seed)
    wrong_table = build_wrong_rule_table(table, diagnostics)

    sources_by_context = defaultdict(set)
    for row in training_rows:
        if row.was_crossroad and row.mu_valid and row.source_id is not None:
            sources_by_context[make_context_key(row.pattern_hash, row.competing_rules)].add(
                row.source_id
            )
    recurring_contexts = sum(len(sources) >= 2 for sources in sources_by_context.values())
    context_rows = []
    for context_key in sorted(diagnostics["per_pattern_status"], key=_context_label):
        context_rows.append(
            {
                "context": _context_label(context_key),
                "competing_rules": "|".join(context_key[1]),
                "distinct_source_circuits": len(sources_by_context.get(context_key, ())),
                "observation_counts": diagnostics["per_pattern_observation_counts"][context_key],
                "source_counts": diagnostics["per_pattern_source_counts"][context_key],
                "average_improvement_vectors": diagnostics["per_pattern_avg_deltas"].get(
                    context_key, {}
                ),
                "status": diagnostics["per_pattern_status"][context_key],
                "learned_rule": table.get(context_key),
            }
        )
    _write_csv(args.out_dir / "context_diagnostics.csv", context_rows)

    print(
        "  training contexts: "
        f"unique={diagnostics['n_patterns_pooled']}, recurring_across_2+_circuits={recurring_contexts}, "
        f"adequately_supported={diagnostics['n_patterns_adequately_supported']}, "
        f"insufficient={diagnostics['n_patterns_insufficient_evidence']}, "
        f"ties={diagnostics['n_patterns_dropped_as_tie']}, learned={len(table)}"
    )

    aggregate = Counter()
    test_contexts = set()
    result_rows = []
    hit_rows = []
    for spec, circuit in test:
        graph = circuit.to_graph()
        result = evaluate_ablation(
            graph,
            spec.circuit_id,
            table,
            null_table,
            null_seed=args.null_seed,
            n_oracle_orderings=args.oracle_orderings,
            diagnostic=False,
        )
        wrong_mu, wrong_outcome, wrong_trace = _terminal_for_table(graph, wrong_table)
        observed = result.learned_lookup_diagnostics.get("observed_context_keys", set())
        test_contexts.update(observed)
        interventions = result.learned_n_differs_from_greedy
        comparisons = {
            "learned_vs_greedy": _comparison(result.learned_mu, result.greedy_mu),
            "learned_vs_null": _comparison(result.learned_mu, result.null_mu),
            "learned_vs_wrong": _comparison(result.learned_mu, wrong_mu),
            "learned_vs_empirical_oracle": _comparison(result.learned_mu, result.oracle_mu),
        }
        for label, comparison in comparisons.items():
            aggregate[f"{label}_{comparison}"] += 1
        aggregate["table_hits"] += result.learned_hits
        aggregate["table_misses"] += result.learned_misses
        aggregate["interventions"] += interventions
        aggregate["null_hits"] += result.null_hits
        aggregate["null_interventions"] += result.null_n_differs_from_greedy
        aggregate["wrong_hits"] += wrong_outcome.n_table_hits
        aggregate["wrong_interventions"] += sum(
            record.differs_from_greedy_choice for record in wrong_trace
        )
        aggregate["empirical_oracle_extraction_failures"] += int(result.oracle_mu is None)
        aggregate["full_reduce_extraction_failures"] += int(result.full_reduce_mu is None)
        for strategy, mu, stop in (
            ("greedy", result.greedy_mu, result.greedy_stop),
            ("learned", result.learned_mu, result.learned_stop),
            ("null", result.null_mu, result.null_stop),
            ("wrong", wrong_mu, wrong_outcome.stop_reason.value),
        ):
            aggregate[f"{strategy}_extraction_failures"] += int(mu is None)
            aggregate[f"{strategy}_{stop}"] += 1
        result_rows.append(
            {
                "circuit_id": spec.circuit_id,
                "family": spec.family,
                "greedy_mu": result.greedy_mu,
                "learned_mu": result.learned_mu,
                "null_mu": result.null_mu,
                "wrong_mu": wrong_mu,
                "empirical_oracle_mu": result.oracle_mu,
                "full_reduce_mu": result.full_reduce_mu,
                **comparisons,
                "table_hits": result.learned_hits,
                "table_misses": result.learned_misses,
                "interventions": interventions,
                "learned_stop": result.learned_stop,
                "greedy_stop": result.greedy_stop,
                "null_stop": result.null_stop,
                "wrong_stop": wrong_outcome.stop_reason.value,
                "wrong_hits": wrong_outcome.n_table_hits,
                "wrong_interventions": sum(r.differs_from_greedy_choice for r in wrong_trace),
                "unique_contexts_on_learned_trajectory": len(observed),
            }
        )
        for table_type, trace in (
            ("learned", result.learned_trace),
            ("null", result.null_trace),
            ("wrong", wrong_trace),
        ):
            for record in trace:
                hit_rows.append(
                    {
                        "circuit_id": spec.circuit_id,
                        "family": spec.family,
                        "table_type": table_type,
                        "step": record.step,
                        "context": _context_label(
                            (record.pattern_hash, record.competing_rules)
                        ),
                        "preferred_rule": record.preferred_rule,
                        "rule_selected": record.rule_selected,
                        "differs_from_greedy_choice": record.differs_from_greedy_choice,
                        "surrogate_before": record.surrogate_before,
                        "surrogate_after": record.surrogate_after,
                    }
                )
    _write_csv(args.out_dir / "held_out_results.csv", result_rows)
    _write_csv(args.out_dir / "held_out_hits.csv", hit_rows)

    train_contexts = set(sources_by_context)
    learned_contexts = set(table)
    overlap = train_contexts & test_contexts
    learned_overlap = learned_contexts & test_contexts
    total_lookups = aggregate["table_hits"] + aggregate["table_misses"]
    summary = {
        "population": {
            "circuits": len(circuits),
            "train": len(train),
            "test": len(test),
            "families": len({spec.family for spec in specs}),
            "training_sweeps": args.training_sweeps,
            "base_seed": args.base_seed,
        },
        "separation_audit": audit,
        "evidence_thresholds": {
            "min_observations_per_rule": args.min_occurrences,
            "min_distinct_source_circuits_per_rule": args.min_source_circuits,
        },
        "training": {
            "crossroad_action_observations": sum(row.was_crossroad for row in training_rows),
            "valid_crossroad_action_observations": sum(
                row.was_crossroad and row.mu_valid for row in training_rows
            ),
            "unique_contexts": diagnostics["n_patterns_pooled"],
            "recurring_contexts_2plus_circuits": recurring_contexts,
            "adequately_supported_contexts": diagnostics["n_patterns_adequately_supported"],
            "insufficient_evidence_contexts": diagnostics["n_patterns_insufficient_evidence"],
            "tied_contexts": diagnostics["n_patterns_dropped_as_tie"],
            "learned_preferences": len(table),
            "stop_reasons": dict(training_stops),
        },
        "held_out": {
            "unique_contexts_on_actual_learned_trajectories": len(test_contexts),
            "train_test_context_overlap": len(overlap),
            "learned_context_overlap": len(learned_overlap),
            "table_hits": aggregate["table_hits"],
            "table_misses": aggregate["table_misses"],
            "table_coverage": aggregate["table_hits"] / total_lookups if total_lookups else 0.0,
            "actual_interventions": aggregate["interventions"],
            "null_hits": aggregate["null_hits"],
            "null_interventions": aggregate["null_interventions"],
            "wrong_hits": aggregate["wrong_hits"],
            "wrong_interventions": aggregate["wrong_interventions"],
            "comparisons": {
                label: {
                    outcome: aggregate[f"{label}_{outcome}"]
                    for outcome in ("win", "tie", "loss", "invalid")
                }
                for label in (
                    "learned_vs_greedy",
                    "learned_vs_null",
                    "learned_vs_wrong",
                    "learned_vs_empirical_oracle",
                )
            },
            "failures_and_stops": {
                f"{strategy}_{metric}": aggregate[f"{strategy}_{metric}"]
                for strategy in ("greedy", "learned", "null", "wrong")
                for metric in (
                    "extraction_failures",
                    "cycle_detected",
                    "hit_step_cap",
                    "all_candidates_failed",
                )
            },
        },
        "ablation_applicable": bool(table),
        "null_keys_changed_from_learned": sum(
            null_table.get(key) != rule for key, rule in table.items()
        ),
        "wrong_rule_keys": len(wrong_table),
    }
    summary["held_out"]["failures_and_stops"].update(
        {
            "empirical_oracle_extraction_failures": aggregate[
                "empirical_oracle_extraction_failures"
            ],
            "full_reduce_extraction_failures": aggregate[
                "full_reduce_extraction_failures"
            ],
        }
    )
    with (args.out_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(
        "  held-out actual trajectories: "
        f"contexts={len(test_contexts)}, train/test overlap={len(overlap)}, "
        f"learned overlap={len(learned_overlap)}, hits={aggregate['table_hits']}, "
        f"interventions={aggregate['interventions']}"
    )
    print(f"  wrote results to {args.out_dir}")


if __name__ == "__main__":
    main()
