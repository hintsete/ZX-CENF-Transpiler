"""Aggregate frozen-protocol natural-population replication runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from scipy.stats import t


COMPARISONS = (
    "learned_vs_greedy",
    "learned_vs_null",
    "learned_vs_wrong",
    "learned_vs_empirical_oracle",
)
OUTCOMES = ("win", "tie", "loss", "invalid")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-runs", type=int, default=5)
    return parser.parse_args()


def _mean_t_interval(values: list[float], confidence: float = 0.95) -> dict:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return {"mean": mean, "low": mean, "high": mean}
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    half_width = t.ppf((1 + confidence) / 2, len(values) - 1) * math.sqrt(
        variance / len(values)
    )
    return {
        "mean": mean,
        "low": max(0.0, mean - half_width),
        "high": min(1.0, mean + half_width),
    }


def main() -> None:
    args = _parse_args()
    paths = sorted(args.root.glob("seed_*/summary.json"))
    if len(paths) != args.expected_runs:
        raise RuntimeError(f"expected {args.expected_runs} summaries, found {len(paths)}")
    summaries = []
    for path in paths:
        with path.open() as handle:
            summaries.append(json.load(handle))

    generator_seeds = []
    fingerprints = []
    for path, summary in zip(paths, summaries):
        with (path.parent / "population_manifest.csv").open(newline="") as handle:
            manifest = list(csv.DictReader(handle))
        generator_seeds.extend(int(row["seed"]) for row in manifest)
        fingerprints.extend(row["fingerprint"] for row in manifest)
    duplicate_generator_seeds = len(generator_seeds) - len(set(generator_seeds))
    duplicate_fingerprints = len(fingerprints) - len(set(fingerprints))
    if duplicate_generator_seeds or duplicate_fingerprints:
        raise RuntimeError(
            "replication populations are not independent: "
            f"duplicate seeds={duplicate_generator_seeds}, "
            f"duplicate fingerprints={duplicate_fingerprints}"
        )

    rows = []
    for summary, summary_path in zip(summaries, paths):
        with (summary_path.parent / "held_out_results.csv").open(newline="") as handle:
            held_out_rows = list(csv.DictReader(handle))
        intervened = [row for row in held_out_rows if int(row["interventions"]) > 0]
        row = {
            "base_seed": summary["population"]["base_seed"],
            "learned_preferences": summary["training"]["learned_preferences"],
            "adequately_supported_contexts": summary["training"][
                "adequately_supported_contexts"
            ],
            "tied_contexts": summary["training"]["tied_contexts"],
            "learned_context_overlap": summary["held_out"]["learned_context_overlap"],
            "table_hits": summary["held_out"]["table_hits"],
            "actual_interventions": summary["held_out"]["actual_interventions"],
            "intervention_circuits": len(intervened),
        }
        for comparison in COMPARISONS:
            for outcome in OUTCOMES:
                row[f"{comparison}_{outcome}"] = summary["held_out"]["comparisons"][
                    comparison
                ][outcome]
        for outcome in OUTCOMES:
            row[f"intervention_learned_vs_greedy_{outcome}"] = sum(
                result["learned_vs_greedy"] == outcome for result in intervened
            )
        rows.append(row)

    totals = {
        comparison: {
            outcome: sum(row[f"{comparison}_{outcome}"] for row in rows)
            for outcome in OUTCOMES
        }
        for comparison in COMPARISONS
    }
    test_circuits_per_run = summaries[0]["population"]["test"]
    rate_intervals = {}
    for comparison in COMPARISONS:
        rate_intervals[comparison] = {
            outcome: _mean_t_interval(
                [row[f"{comparison}_{outcome}"] / test_circuits_per_run for row in rows]
            )
            for outcome in OUTCOMES
        }

    failure_totals = Counter()
    training_stop_totals = Counter()
    for summary in summaries:
        failure_totals.update(summary["held_out"]["failures_and_stops"])
        training_stop_totals.update(summary["training"]["stop_reasons"])

    aggregate = {
        "protocol_frozen": {
            "runs": len(summaries),
            "population_seeds": [row["base_seed"] for row in rows],
            "circuits_per_run": summaries[0]["population"]["circuits"],
            "train_per_run": summaries[0]["population"]["train"],
            "test_per_run": test_circuits_per_run,
            "min_observations_per_rule": summaries[0]["evidence_thresholds"][
                "min_observations_per_rule"
            ],
            "min_source_circuits_per_rule": summaries[0]["evidence_thresholds"][
                "min_distinct_source_circuits_per_rule"
            ],
            "duplicate_generator_seeds_across_populations": duplicate_generator_seeds,
            "duplicate_circuit_fingerprints_across_populations": duplicate_fingerprints,
        },
        "totals": {
            "learned_preferences": sum(row["learned_preferences"] for row in rows),
            "adequately_supported_contexts": sum(
                row["adequately_supported_contexts"] for row in rows
            ),
            "tied_contexts": sum(row["tied_contexts"] for row in rows),
            "learned_context_overlap": sum(row["learned_context_overlap"] for row in rows),
            "table_hits": sum(row["table_hits"] for row in rows),
            "actual_interventions": sum(row["actual_interventions"] for row in rows),
            "intervention_circuits": sum(row["intervention_circuits"] for row in rows),
            "intervention_circuit_learned_vs_greedy": {
                outcome: sum(
                    row[f"intervention_learned_vs_greedy_{outcome}"] for row in rows
                )
                for outcome in OUTCOMES
            },
            "comparisons": totals,
            "training_stop_reasons": dict(training_stop_totals),
            "held_out_failures_and_stops": dict(failure_totals),
        },
        "per_run_rate_95pct_t_intervals": rate_intervals,
        "per_run": rows,
    }
    with (args.root / "replication_summary.json").open("w") as handle:
        json.dump(aggregate, handle, indent=2, sort_keys=True)
    with (args.root / "replication_runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
