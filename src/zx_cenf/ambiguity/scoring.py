"""Ambiguity scoring on PyZX simplification: extracts a circuit after each
randomized-order run and compares gate count / T-count / depth/ TwoQubit count """

from __future__ import annotations

from dataclasses import dataclass, field

import pyzx as zx

from zx_cenf.ambiguity.pyzx_ordering import run_full_reduce_baseline, run_randomized_pass_order


@dataclass
class OrderingRunResult:
    seed: int
    spider_count: int
    edge_count: int
    tcount: int | None
    twoqubitcount: int | None
    depth: int | None
    extraction_failed: bool
    pass_counts: dict


@dataclass
class AmbiguityResult:
    diagram_id: str
    runs: list[OrderingRunResult]
    full_reduce_baseline: OrderingRunResult

    @property
    def successful_runs(self) -> list[OrderingRunResult]:
        return [r for r in self.runs if not r.extraction_failed]

    def _metric_values(self, metric: str) -> list[int]:
        return [getattr(r, metric) for r in self.successful_runs if getattr(r, metric) is not None]
    

    def metric_summary(self, metric: str = "twoqubitcount") -> dict | None:
        """Return a human-readable summary of ambiguity for a metric."""
        values = self._metric_values(metric)
        if len(values) < 2:
            return None

        mean = sum(values) / len(values)
        score = 0.0 if mean == 0 else (max(values) - min(values)) / mean

        return {
            "min": min(values),
            "max": max(values),
            "mean": round(mean, 1),
            "n": len(values),
            "ambiguity_pct": round(score * 100, 1),
            "label": self._label_ambiguity(score),
        }

    @staticmethod
    def _label_ambiguity(score: float) -> str:
        """Qualitative interpretation of ambiguity magnitude."""
        if score == 0:
            return "none"
        if score < 0.10:
            return "negligible"
        if score < 0.25:
            return "moderate"
        if score < 0.50:
            return "substantial"
        return "large"

    def ambiguity_score(self, metric: str = "twoqubitcount") -> float | None:
        """(max - min) / mean for the given circuit metric across sampled
        pass orderings. None if too few successful extractions to compare."""
        values = self._metric_values(metric)
        if len(values) < 2:
            return None
        mean = sum(values) / len(values)
        return 0.0 if mean == 0 else (max(values) - min(values)) / mean

    def is_ambiguous(self, metric: str = "twoqubitcount") -> bool | None:
        values = self._metric_values(metric)
        if len(values) < 2:
            return None
        return len(set(values)) > 1

    def headroom_vs_full_reduce(self, metric: str = "twoqubitcount") -> float | None:
        """Compares how much better(or worse) the BEST randomized ordering vs
        PyZX's own full_reduce heuristic"""
        values = self._metric_values(metric)
        baseline = getattr(self.full_reduce_baseline, metric)
        if not values or baseline is None:
            return None
        best = min(values)
        return (baseline - best) / baseline if baseline != 0 else None
    


def _extract_and_measure(g, seed: int, pass_counts: dict) -> OrderingRunResult:
    spider_count = g.num_vertices()
    edge_count = g.num_edges()
    try:
        c = zx.extract_circuit(g.copy(), quiet=True)
        return OrderingRunResult(
            seed=seed,
            spider_count=spider_count,
            edge_count=edge_count,
            tcount=c.tcount(),
            twoqubitcount=c.twoqubitcount(),
            depth=c.depth(),
            extraction_failed=False,
            pass_counts=pass_counts,
        )
    except Exception:
        # Extraction can legitimately fail on some intermediate graph
     
        return OrderingRunResult(
            seed=seed,
            spider_count=spider_count,
            edge_count=edge_count,
            tcount=None,
            twoqubitcount=None,
            depth=None,
            extraction_failed=True,
            pass_counts=pass_counts,
        )


def score_diagram(g, diagram_id: str, n_orderings: int = 30, base_seed: int = 0) -> AmbiguityResult:
    runs = []
    for i in range(n_orderings):
        seed = base_seed + i
        g_copy = g.copy()
        pass_counts = run_randomized_pass_order(g_copy, seed=seed)
        runs.append(_extract_and_measure(g_copy, seed, pass_counts))

    g_baseline = g.copy()
    run_full_reduce_baseline(g_baseline)
    baseline_result = _extract_and_measure(g_baseline, seed=-1, pass_counts={})

    return AmbiguityResult(diagram_id, runs, baseline_result)
