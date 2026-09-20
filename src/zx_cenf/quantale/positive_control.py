"""Synthetic positive-control population for the Phase F mechanism test.

The control starts from a circuit-derived ZX decision state containing a
reproducible ``pivot`` versus ``pivot_gadget`` crossroad.  Choosing
``pivot_gadget`` improves terminal two-qubit count by three under the Phase F
greedy continuation, while increasing depth.  That makes it a deliberate test
of the declared lexicographic priority rather than a claim about naturally
occurring circuit populations.
"""

from __future__ import annotations

import random

import pyzx as zx
import pyzx.simplify as simp

from zx_cenf.quantale.instrumented_reduction import (
    DEFAULT_CONTEXT_REPRESENTATION,
    _counterfactual_crossroad_rows,
    compute_context_hash_variants,
    decision_context_vertices,
    enumerate_all_candidates,
    local_competing_rules,
    safe_apply_match,
)
from zx_cenf.quantale.pattern_hash import make_context_key


MOTIF_CIRCUIT_SEED = 10
MOTIF_EXPLORATION_SEED = 0
MOTIF_REWRITE_PREFIX_STEPS = 11

TARGET_PATTERN_HASH = "7df5c8ce565cfcc266b52ae025ca9de7"
TARGET_COMPETING_RULES = ("pivot", "pivot_gadget")
TARGET_CONTEXT_KEY = make_context_key(
    TARGET_PATTERN_HASH,
    TARGET_COMPETING_RULES,
)
EXPECTED_PREFERRED_RULE = "pivot_gadget"
INTENTIONALLY_WRONG_RULE = "pivot"


def build_positive_control_motif_graph():
    """Reproduce the circuit-derived graph immediately before the crossroad."""
    circuit = zx.generate.CNOT_HAD_PHASE_circuit(
        3,
        20,
        p_had=0.25,
        p_t=0.35,
        clifford=False,
        seed=MOTIF_CIRCUIT_SEED,
    )
    graph = circuit.to_graph()
    simp.to_gh(graph)
    rng = random.Random(MOTIF_EXPLORATION_SEED)

    n_applied = 0
    while n_applied < MOTIF_REWRITE_PREFIX_STEPS:
        candidates = enumerate_all_candidates(graph)
        if not candidates:
            if not simp.gadget_simp(graph):
                raise RuntimeError(
                    "positive-control motif reached a fixed point before "
                    "the expected decision state"
                )
            continue

        remaining = list(candidates)
        rng.shuffle(remaining)
        applied = False
        while remaining and not applied:
            applied = safe_apply_match(graph, *remaining.pop())

        if not applied:
            raise RuntimeError(
                "positive-control motif could not replay its rewrite prefix"
            )

        graph.remove_isolated_vertices()
        n_applied += 1

    _find_target_crossroad(graph)
    return graph


def build_positive_control_graph(index: int, split: str):
    """Embed the fixed motif beside an independently generated background."""
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    if index < 0:
        raise ValueError("index must be non-negative")

    seed_base = 1_000 if split == "train" else 100_000
    background = zx.generate.CNOT_HAD_PHASE_circuit(
        2 + index % 3,
        5 + index % 8,
        p_had=0.30,
        p_t=0.40,
        clifford=False,
        seed=seed_base + index,
    ).to_graph()

    graph = build_positive_control_motif_graph().tensor(background)
    simp.to_gh(graph)
    _find_target_crossroad(graph)
    return graph


def collect_target_action_observations(graph) -> list:
    """Evaluate every action at the designated control crossroad once."""
    work = graph.clone()
    simp.to_gh(work)
    candidates, chosen, context_hashes = _find_target_crossroad(work)
    rows = _counterfactual_crossroad_rows(
        work,
        candidates,
        chosen,
        context_hashes,
    )

    observed_rules = {row.rule for row in rows}
    if observed_rules != set(TARGET_COMPETING_RULES):
        raise RuntimeError(
            "positive-control action set changed: "
            f"expected {TARGET_COMPETING_RULES}, got {sorted(observed_rules)}"
        )
    if not all(row.mu_valid for row in rows):
        raise RuntimeError("positive-control terminal rollout was not extractable")

    return rows


def _find_target_crossroad(graph):
    candidates = enumerate_all_candidates(graph)
    observed_keys = set()

    for candidate in candidates:
        competing_rules = local_competing_rules(candidates, candidate)
        if len(competing_rules) < 2:
            continue

        context_vertices = decision_context_vertices(
            graph,
            candidates,
            candidate,
        )
        context_hashes = compute_context_hash_variants(
            graph,
            context_vertices,
        )
        context_key = make_context_key(
            context_hashes[DEFAULT_CONTEXT_REPRESENTATION],
            competing_rules,
        )
        observed_keys.add(context_key)

        if context_key == TARGET_CONTEXT_KEY:
            return candidates, candidate, context_hashes

    raise RuntimeError(
        "expected positive-control crossroad was not found; "
        f"observed {sorted(observed_keys)!r}"
    )