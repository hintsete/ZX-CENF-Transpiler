"""Crossroad-aware instrumented reduction (TRAINING)."""

from __future__ import annotations

import random
from dataclasses import dataclass

import pyzx.simplify as simp

from zx_cenf.quantale.pattern_hash import compute_local_pattern_hash
from zx_cenf.quantale.valuation import mu_from_pyzx_graph


INSTRUMENTABLE_RULES = {
    "spider_fusion": simp.fuse_simp,
    "pivot": simp.pivot_simp,
    "lcomp": simp.lcomp_simp,
    "pivot_gadget": simp.pivot_gadget_simp,
    "pivot_boundary": simp.pivot_boundary_simp,
    "id_removal": simp.id_simp,
}


@dataclass
class RewriteLogRow:
    pattern_hash: str
    rule: str
    was_crossroad: bool
    competing_rules: tuple = ()
    mu_before: tuple | None = None
    mu_after: tuple | None = None
    mu_valid: bool = True
    context_hash_variants: dict | None = None


def enumerate_all_candidates(g) -> list:
    candidates = []

    for rule_name, rewrite_obj in INSTRUMENTABLE_RULES.items():
        for match in rewrite_obj.find_all_matches(g):
            candidates.append(
                (
                    rule_name,
                    match,
                )
            )

    return candidates


def match_vertices(match) -> list:
    if isinstance(match, tuple):
        return list(match)

    return [match]


def local_competing_rules(
    candidates,
    chosen,
) -> tuple:
    _chosen_rule, chosen_match = chosen

    chosen_vs = set(
        match_vertices(chosen_match)
    )

    rules = set()

    for rule_name, match in candidates:
        candidate_vs = set(
            match_vertices(match)
        )

        if chosen_vs & candidate_vs:
            rules.add(rule_name)

    return tuple(sorted(rules))


def competing_candidates(
    candidates,
    chosen,
) -> list:
    _chosen_rule, chosen_match = chosen

    chosen_vs = set(
        match_vertices(chosen_match)
    )

    local_candidates = []

    for rule_name, match in candidates:
        candidate_vs = set(
            match_vertices(match)
        )

        if chosen_vs & candidate_vs:
            local_candidates.append(
                (
                    rule_name,
                    match,
                )
            )

    return local_candidates


def decision_context_vertices(
    g,
    candidates,
    chosen,
) -> list:
    """
    Shared-anchor decision context: pairwise intersections among the
    locally-competing redexes, no hop expansion.

    Falls back to the first local redex only if no pair shares a vertex
    (should be rare, since every local candidate overlaps the chosen one
    by construction).
    """
    local_candidates = competing_candidates(
        candidates,
        chosen,
    )

    if not local_candidates:
        return []

    redex_sets = [
        set(match_vertices(match))
        for _rule_name, match in local_candidates
    ]

    shared_anchors = set()

    for i in range(len(redex_sets)):
        for j in range(i + 1, len(redex_sets)):
            shared_anchors.update(
                redex_sets[i] & redex_sets[j]
            )

    if not shared_anchors:
        shared_anchors.update(
            redex_sets[0]
        )

    return sorted(shared_anchors)


def compute_context_hash_variants(
    g,
    context_vertices,
) -> dict:
    return {
        "FULL": compute_local_pattern_hash(
            g,
            context_vertices,
            include_degree=True,
            include_phase=True,
        ),
        "NO_DEGREE": compute_local_pattern_hash(
            g,
            context_vertices,
            include_degree=False,
            include_phase=True,
        ),
        "NO_PHASE": compute_local_pattern_hash(
            g,
            context_vertices,
            include_degree=True,
            include_phase=False,
        ),
        "STRUCTURAL": compute_local_pattern_hash(
            g,
            context_vertices,
            include_degree=False,
            include_phase=False,
        ),
    }


def print_crossroad_diagnostic(
    g,
    candidates,
    chosen,
) -> None:
    local_candidates = competing_candidates(
        candidates,
        chosen,
    )

    distinct_rules = sorted(
        {
            rule_name
            for rule_name, _match in local_candidates
        }
    )

    if len(distinct_rules) < 2:
        return

    chosen_rule, _chosen_match = chosen

    results = []

    for rule_name, match in local_candidates:
        vertices = match_vertices(match)

        pattern_hash = compute_local_pattern_hash(
            g,
            vertices,
        )

        results.append(
            (
                rule_name,
                vertices,
                pattern_hash,
            )
        )

    hashes = {
        pattern_hash
        for _r, _v, pattern_hash in results
    }

    print("\n" + "=" * 80)
    print("CROSSROAD HASH DIAGNOSTIC")
    print(
        f"competing rules: {tuple(distinct_rules)}"
    )
    print(
        f"chosen rule:     {chosen_rule}"
    )

    for rule_name, vertices, pattern_hash in results:
        print(
            f"  {rule_name:18s} "
            f"redex={vertices!r} "
            f"hash={pattern_hash}"
        )

    print(
        "  SAME HASH FOR ALL CANDIDATES: "
        f"{'YES' if len(hashes) == 1 else 'NO'}"
    )

    print("=" * 80)


def print_decision_context_diagnostic(
    g,
    candidates,
    chosen,
    context_vertices,
    context_hashes,
) -> None:
    local_candidates = competing_candidates(
        candidates,
        chosen,
    )

    distinct_rules = sorted(
        {
            rule_name
            for rule_name, _match in local_candidates
        }
    )

    if len(distinct_rules) < 2:
        return

    print("DECISION-CONTEXT DIAGNOSTIC")

    print(
        f"  competing rules: {tuple(distinct_rules)}"
    )

    print(
        f"  context vertices: {context_vertices!r}"
    )

    print(
        f"  FULL:        {context_hashes['FULL']}"
    )

    print(
        f"  NO_DEGREE:   {context_hashes['NO_DEGREE']}"
    )

    print(
        f"  NO_PHASE:    {context_hashes['NO_PHASE']}"
    )

    print(
        f"  STRUCTURAL:  {context_hashes['STRUCTURAL']}"
    )

    print("=" * 80)


def apply_match(
    g,
    rule_name: str,
    match,
) -> None:
    rewrite_obj = INSTRUMENTABLE_RULES[
        rule_name
    ]

    if isinstance(match, tuple):
        rewrite_obj.applier(
            g,
            match[0],
            match[1],
        )
    else:
        rewrite_obj.applier(
            g,
            match,
        )


def safe_apply_match(
    g,
    rule_name: str,
    match,
) -> bool:
    try:
        apply_match(
            g,
            rule_name,
            match,
        )

        return True

    except Exception:
        return False


def run_instrumented_reduction(
    g,
    seed: int,
    max_steps: int = 200,
    compute_mu_per_step: bool = True,
    verbose: bool = False,
) -> tuple:
    rng = random.Random(seed)

    simp.to_gh(g)

    log_rows: list[RewriteLogRow] = []

    reached_fixed_point = False

    for _step in range(max_steps):
        candidates = enumerate_all_candidates(g)

        if not candidates:
            if not simp.gadget_simp(g):
                reached_fixed_point = True
                break

            continue

        remaining = list(candidates)

        rng.shuffle(remaining)

        applied_ok = False

        while remaining and not applied_ok:
            chosen = remaining.pop()

            rule_name, match = chosen

            competing = local_competing_rules(
                candidates,
                chosen,
            )

            was_crossroad = len(competing) > 1

            context_vertices = decision_context_vertices(
                g,
                candidates,
                chosen,
            )

            pattern_hash = compute_local_pattern_hash(
                g,
                context_vertices,
            )

            context_hashes = None

            if was_crossroad:
                context_hashes = compute_context_hash_variants(
                    g,
                    context_vertices,
                )

                if verbose:
                    print_crossroad_diagnostic(
                        g,
                        candidates,
                        chosen,
                    )

                    print_decision_context_diagnostic(
                        g,
                        candidates,
                        chosen,
                        context_vertices,
                        context_hashes,
                    )

            mu_before = None
            mu_after = None
            mu_valid = True

            if compute_mu_per_step:
                try:
                    mu_before = mu_from_pyzx_graph(
                        g
                    ).coords

                except Exception:
                    mu_valid = False

            applied_ok = safe_apply_match(
                g,
                rule_name,
                match,
            )

        if not applied_ok:
            break

        g.remove_isolated_vertices()

        if compute_mu_per_step and mu_valid:
            try:
                mu_after = mu_from_pyzx_graph(
                    g
                ).coords

            except Exception:
                mu_valid = False

        log_rows.append(
            RewriteLogRow(
                pattern_hash=pattern_hash,
                rule=rule_name,
                was_crossroad=was_crossroad,
                competing_rules=competing,
                mu_before=mu_before,
                mu_after=mu_after,
                mu_valid=mu_valid,
                context_hash_variants=context_hashes,
            )
        )

    return (
        log_rows,
        reached_fixed_point,
    )