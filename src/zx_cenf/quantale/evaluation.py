"""Held-out evaluation, and the causal ablation."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cmp_to_key

from zx_cenf.ambiguity.pyzx_ordering import run_randomized_pass_order, run_full_reduce_baseline
from zx_cenf.quantale.instrumented_reduction import (
    enumerate_all_candidates,
    local_competing_rules,
)
from zx_cenf.quantale.pattern_hash import compute_local_pattern_hash, is_valid_pattern
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
    test_pattern_hash_variants: dict = field(default_factory=dict)


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


def collect_test_pattern_hash_variants(g_original, max_steps: int = 200) -> dict:
    """Collect the four representation hashes actually encountered by
    the held-out diagram during a deterministic greedy-style reduction,
    using the SAME shared-anchor decision-context definition as
    training (not candidate-centered hashing)."""
    import pyzx.simplify as simp
    from zx_cenf.quantale.instrumented_reduction import (
        safe_apply_match,
        apply_match,
        decision_context_vertices,
        compute_context_hash_variants,
    )

    g = g_original.copy()
    simp.to_gh(g)
    hashes = {"FULL": set(), "NO_DEGREE": set(), "NO_PHASE": set(), "STRUCTURAL": set()}

    for _step in range(max_steps):
        candidates = enumerate_all_candidates(g)
        if not candidates:
            if not simp.gadget_simp(g):
                break
            continue

        chosen = None
        for candidate in candidates:
            rule_name, match = candidate
            test_graph = g.copy()
            try:
                apply_match(test_graph, rule_name, match)
                chosen = candidate
                break
            except Exception:
                continue
        if chosen is None:
            break

        competing = local_competing_rules(candidates, chosen)
        if len(competing) > 1:
            context_vertices = decision_context_vertices(g, candidates, chosen)
            context_hashes = compute_context_hash_variants(g, context_vertices)
            for representation, pattern_hash in context_hashes.items():
                if pattern_hash and pattern_hash != "ERROR":
                    hashes[representation].add(pattern_hash)

        rule_name, match = chosen
        if not safe_apply_match(g, rule_name, match):
            break
        g.remove_isolated_vertices()

    return hashes


def collect_test_pattern_hashes(g_original, max_steps: int = 200) -> list:
    variants = collect_test_pattern_hash_variants(g_original, max_steps=max_steps)
    return sorted(variants["FULL"])


def evaluate_on_held_out_diagram(
    g_original,
    diagram_id: str,
    pattern_table: dict,
    n_oracle_orderings: int = 30,
) -> EvaluationResult:
    oracle_mu = compute_empirical_oracle(g_original, n_oracle_orderings)

    g_greedy = g_original.copy()
    greedy_outcome = naive_greedy_reduce(g_greedy)
    finalize_for_extraction(g_greedy)
    greedy_mu = _safe_mu(g_greedy)

    g_table = g_original.copy()
    # table_outcome, _trace = table_guided_reduce(g_table, pattern_table, )
    table_outcome, _trace = table_guided_reduce(
    g_table,
    pattern_table,
    diagnostic=True,
)
    finalize_for_extraction(g_table)
    table_mu = _safe_mu(g_table)

    g_full = g_original.copy()
    run_full_reduce_baseline(g_full)
    full_reduce_mu = _safe_mu(g_full)

    test_pattern_hash_variants = collect_test_pattern_hash_variants(g_original)

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
        test_pattern_hashes=sorted(test_pattern_hash_variants["FULL"]),
        test_pattern_hash_variants=test_pattern_hash_variants,
    )


def evaluate_ablation(
    g_original,
    diagram_id: str,
    learned_table: dict,
    null_table: dict,
    null_seed: int,
    n_oracle_orderings: int = 30,
) -> AblationResult:
    oracle_mu = compute_empirical_oracle(g_original, n_oracle_orderings)

    g_greedy = g_original.copy()
    greedy_outcome = naive_greedy_reduce(g_greedy)
    finalize_for_extraction(g_greedy)
    greedy_mu = _safe_mu(g_greedy)

    g_learned = g_original.copy()
    # learned_outcome, learned_trace = table_guided_reduce(g_learned, learned_table)
    learned_outcome, learned_trace = table_guided_reduce(
    g_learned,
    learned_table,
    diagnostic=True,
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
    )