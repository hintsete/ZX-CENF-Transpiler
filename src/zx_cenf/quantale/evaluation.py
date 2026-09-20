"""Held-out evaluation, and the causal ablation."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cmp_to_key

from zx_cenf.ambiguity.pyzx_ordering import run_randomized_pass_order, run_full_reduce_baseline
from zx_cenf.quantale.strategies import (
    PRIORITY,
    finalize_for_extraction,
    naive_greedy_reduce,
    table_guided_reduce,
)
from zx_cenf.quantale.valuation import mu_from_pyzx_graph


@dataclass
class EvaluationResult:
    diagram_id: str
    oracle_mu: tuple | None
    naive_greedy_mu: tuple | None
    naive_greedy_stop: str
    table_guided_mu: tuple | None
    table_guided_stop: str
    table_hits: int
    table_misses: int
    full_reduce_mu: tuple | None
    test_pattern_hashes: list = field(default_factory=list)
    test_context_keys: list = field(default_factory=list)
    table_n_differs_from_greedy: int = 0
    table_trace: list = field(default_factory=list)


@dataclass
class AblationResult:
    diagram_id: str
    oracle_mu: tuple | None
    greedy_mu: tuple | None
    greedy_stop: str
    learned_mu: tuple | None
    learned_stop: str
    learned_hits: int
    learned_misses: int
    learned_n_differs_from_greedy: int
    learned_trace: list = field(default_factory=list)
    null_mu: tuple | None = None
    null_stop: str = ""
    null_hits: int = 0
    null_misses: int = 0
    null_n_differs_from_greedy: int = 0
    null_trace: list = field(default_factory=list)
    full_reduce_mu: tuple | None = None
    null_seed: int = 0
    learned_lookup_diagnostics: dict = field(default_factory=dict)
    null_lookup_diagnostics: dict = field(default_factory=dict)


def _safe_mu(g):
    try:
        return mu_from_pyzx_graph(g).coords
    except Exception:
        return None


def _mu_lexicographically_better(a, b, tol: float = 1e-9) -> int:
    for coord in PRIORITY:
        if a[coord] < b[coord] - tol:
            return 1
        if b[coord] < a[coord] - tol:
            return -1
    return 0


def compute_empirical_oracle(g_original, n_orderings: int = 30, base_seed: int = 0):
    results = []
    for i in range(n_orderings):
        g_copy = g_original.copy()
        run_randomized_pass_order(g_copy, seed=base_seed + i)
        mu = _safe_mu(g_copy)
        if mu is not None:
            results.append(mu)
    if not results:
        return None
    results.sort(key=cmp_to_key(_mu_lexicographically_better), reverse=True)
    return results[0]


def evaluate_on_held_out_diagram(
    g_original,
    diagram_id: str,
    pattern_table: dict,
    n_oracle_orderings: int = 30,
    diagnostic: bool = True,
) -> EvaluationResult:
    oracle_mu = compute_empirical_oracle(g_original, n_oracle_orderings)

    g_greedy = g_original.copy()
    greedy_outcome = naive_greedy_reduce(g_greedy)
    finalize_for_extraction(g_greedy)
    greedy_mu = _safe_mu(g_greedy)

    g_table = g_original.copy()
    table_outcome, table_trace = table_guided_reduce(
        g_table,
        pattern_table,
        diagnostic=diagnostic,
    )
    finalize_for_extraction(g_table)
    table_mu = _safe_mu(g_table)

    g_full = g_original.copy()
    run_full_reduce_baseline(g_full)
    full_reduce_mu = _safe_mu(g_full)

    lookup_diagnostics = table_outcome.lookup_diagnostics
    test_pattern_hashes = sorted(lookup_diagnostics["observed_hashes"])
    test_context_keys = sorted(lookup_diagnostics["observed_context_keys"])

    return EvaluationResult(
        diagram_id=diagram_id,
        oracle_mu=oracle_mu,
        naive_greedy_mu=greedy_mu,
        naive_greedy_stop=greedy_outcome.stop_reason.value,
        table_guided_mu=table_mu,
        table_guided_stop=table_outcome.stop_reason.value,
        table_hits=table_outcome.n_table_hits,
        table_misses=table_outcome.n_table_misses,
        full_reduce_mu=full_reduce_mu,
        test_pattern_hashes=test_pattern_hashes,
        test_context_keys=test_context_keys,
        table_n_differs_from_greedy=sum(
            1 for row in table_trace if row.differs_from_greedy_choice
        ),
        table_trace=table_trace,
    )


def evaluate_ablation(
    g_original,
    diagram_id: str,
    learned_table: dict,
    null_table: dict,
    null_seed: int,
    n_oracle_orderings: int = 30,
    diagnostic: bool = True,
) -> AblationResult:
    oracle_mu = compute_empirical_oracle(g_original, n_oracle_orderings)

    g_greedy = g_original.copy()
    greedy_outcome = naive_greedy_reduce(g_greedy)
    finalize_for_extraction(g_greedy)
    greedy_mu = _safe_mu(g_greedy)

    g_learned = g_original.copy()
    learned_outcome, learned_trace = table_guided_reduce(
        g_learned,
        learned_table,
        diagnostic=diagnostic,
    )
    finalize_for_extraction(g_learned)
    learned_mu = _safe_mu(g_learned)

    g_null = g_original.copy()
    null_outcome, null_trace = table_guided_reduce(g_null, null_table)
    finalize_for_extraction(g_null)
    null_mu = _safe_mu(g_null)

    g_full = g_original.copy()
    run_full_reduce_baseline(g_full)
    full_reduce_mu = _safe_mu(g_full)

    return AblationResult(
        diagram_id=diagram_id,
        oracle_mu=oracle_mu,
        greedy_mu=greedy_mu,
        greedy_stop=greedy_outcome.stop_reason.value,
        learned_mu=learned_mu,
        learned_stop=learned_outcome.stop_reason.value,
        learned_hits=learned_outcome.n_table_hits,
        learned_misses=learned_outcome.n_table_misses,
        learned_n_differs_from_greedy=sum(1 for r in learned_trace if r.differs_from_greedy_choice),
        learned_trace=learned_trace,
        null_mu=null_mu,
        null_stop=null_outcome.stop_reason.value,
        null_hits=null_outcome.n_table_hits,
        null_misses=null_outcome.n_table_misses,
        null_n_differs_from_greedy=sum(1 for r in null_trace if r.differs_from_greedy_choice),
        null_trace=null_trace,
        full_reduce_mu=full_reduce_mu,
        null_seed=null_seed,
        learned_lookup_diagnostics=learned_outcome.lookup_diagnostics,
        null_lookup_diagnostics=null_outcome.lookup_diagnostics,
    )
