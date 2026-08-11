"""Track 1: Randomizes the order of non-confluent PyZX simplification
passes to explore the ambiguity of the ZX-calculus rewrite system.  

"""

from __future__ import annotations

import random

import pyzx as zx
import pyzx.simplify as simp


CLEANUP_PASSES = {
    "spider": simp.spider_simp,
    "id": simp.id_simp,
}


NONCONFLUENT_PASSES = {
    "pivot": simp.pivot_simp,
    "lcomp": simp.lcomp_simp,
    "pivot_gadget": simp.pivot_gadget_simp,
    "pivot_boundary": simp.pivot_boundary_simp,
    "gadget": simp.gadget_simp,
}

ALL_PASSES = {**CLEANUP_PASSES, **NONCONFLUENT_PASSES}


def run_randomized_pass_order(g, seed: int, max_rounds: int = 50) -> dict:
    """Repeatedly shuffle pass order and run each pass once (each pass
    internally fixpoints itself), until no pass in a full round makes
    any change. Mutates `g` in place.

    Returns per-pass application counts for this run.
    """
    rng = random.Random(seed)
    simp.to_gh(g)

    counts = {name: 0 for name in ALL_PASSES}
    for _ in range(max_rounds):
        order = list(ALL_PASSES.keys())
        rng.shuffle(order)
        any_applied = False
        for name in order:
            applied = ALL_PASSES[name](g)  
            if applied:
                counts[name] += 1
                any_applied = True
        if not any_applied:
            break

    g.remove_isolated_vertices()
    return counts


def run_full_reduce_baseline(g) -> None:
    """PyZX's own fixed heuristic order """
    simp.full_reduce(g, quiet=True)