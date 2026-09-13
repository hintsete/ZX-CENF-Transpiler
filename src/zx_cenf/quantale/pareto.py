"""Offline Pareto and correlation analysis."""

from __future__ import annotations

import itertools
import statistics
from dataclasses import dataclass, field
from typing import Any

from zx_cenf.quantale.algebra import QuantaleValue
from zx_cenf.quantale.valuation import COORDINATE_NAMES, DIM, mu_from_pyzx_graph


@dataclass
class DiagramValuations:
    diagram_id: str
    n_orderings: int
    values: list = field(default_factory=list)          # list[QuantaleValue], successful runs only
    n_extraction_failures: int = 0
    failure_seeds: list = field(default_factory=list)
    success_seeds: list = field(default_factory=list)


def collect_diagram_valuations(
    diagram_id: str,
    run_data: list[tuple[int, Any, dict]],
) -> DiagramValuations:
  
    result = DiagramValuations(diagram_id=diagram_id, n_orderings=len(run_data))

    for seed, g, _pass_counts in run_data:
        try:
            value = mu_from_pyzx_graph(g)
            result.values.append(value)
            result.success_seeds.append(seed)
        except Exception:
            result.n_extraction_failures += 1
            result.failure_seeds.append(seed)

    return result


def pareto_frontier(values: list) -> list:
    frontier = []
    for i, v in enumerate(values):
        dominated = any(other.dominates(v) for j, other in enumerate(values) if j != i)
        if not dominated:
            frontier.append(i)
    return frontier


def comparability_ratio(values: list):

    n = len(values)
    if n < 2:
        return None
    pairs = list(itertools.combinations(range(n), 2))
    comparable = sum(1 for i, j in pairs if values[i].comparable_to(values[j]))
    return comparable / len(pairs)


def coordinate_correlations(values: list) -> dict:
    n = len(values)
    correlations = {}
    for i, j in itertools.combinations(range(DIM), 2):
        xi = [v.coords[i] for v in values]
        xj = [v.coords[j] for v in values]
        key = f"{COORDINATE_NAMES[i]}_vs_{COORDINATE_NAMES[j]}"
        if n < 2 or len(set(xi)) < 2 or len(set(xj)) < 2:
            correlations[key] = None
            continue
        try:
            correlations[key] = statistics.correlation(xi, xj)
        except statistics.StatisticsError:
            correlations[key] = None
    return correlations


def extraction_failure_bias_check(dv: DiagramValuations) -> dict:
    total = dv.n_orderings
    n_fail = dv.n_extraction_failures
    if total == 0:
        return {
            "n_failures": 0,
            "failure_rate": None,
            "first_half_failures": None,
            "second_half_failures": None,
        }

    all_seeds = dv.success_seeds + dv.failure_seeds
    midpoint = (min(all_seeds) + max(all_seeds)) / 2 if all_seeds else None
    first_half = sum(1 for s in dv.failure_seeds if midpoint is not None and s < midpoint)
    second_half = n_fail - first_half if midpoint is not None else None

    return {
        "n_failures": n_fail,
        "failure_rate": round(n_fail / total, 3),
        "first_half_failures": first_half if midpoint is not None else None,
        "second_half_failures": second_half,
    }


def summarize_diagram(dv: DiagramValuations) -> dict:
    frontier_idx = pareto_frontier(dv.values)
    row = {
        "diagram_id": dv.diagram_id,
        "n_orderings": dv.n_orderings,
        "n_successful": len(dv.values),
        "n_extraction_failures": dv.n_extraction_failures,
        "pareto_frontier_size": len(frontier_idx),
        "pareto_frontier_fraction": round(len(frontier_idx) / len(dv.values), 3) if dv.values else None,
        "comparability_ratio": comparability_ratio(dv.values),
    }
    row.update(coordinate_correlations(dv.values))
    row.update({f"bias_{k}": v for k, v in extraction_failure_bias_check(dv).items()})
    return row