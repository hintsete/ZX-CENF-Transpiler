"""Crossroad-aware instrumented reduction (TRAINING)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

import pyzx.simplify as simp

from zx_cenf.quantale.pattern_hash import (
    compute_graph_state_hash,
    compute_local_pattern_hash,
)
from zx_cenf.quantale.valuation import mu_from_pyzx_graph


INSTRUMENTABLE_RULES = {
    "spider_fusion": simp.fuse_simp,
    "pivot": simp.pivot_simp,
    "lcomp": simp.lcomp_simp,
    "pivot_gadget": simp.pivot_gadget_simp,
    "pivot_boundary": simp.pivot_boundary_simp,
    "id_removal": simp.id_simp,
}

DEFAULT_CONTEXT_REPRESENTATION = "NO_DEGREE"


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
    selected_on_exploration_path: bool = True
    reward_source: str = "terminal_greedy_rollout"
    source_id: str | None = None


class TrainingStopReason(str, Enum):
    STRUCTURAL_FIXED_POINT = "structural_fixed_point"
    CYCLE_DETECTED = "cycle_detected"
    HIT_STEP_CAP = "hit_step_cap"
    ALL_CANDIDATES_FAILED = "all_candidates_failed"


@dataclass(frozen=True)
class TrainingOutcome:
    stop_reason: TrainingStopReason
    n_steps: int

    @property
    def reached_fixed_point(self) -> bool:
        return self.stop_reason == TrainingStopReason.STRUCTURAL_FIXED_POINT


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
    return tuple(
        sorted(
            {
                rule_name
                for rule_name, _match in competing_candidates(candidates, chosen)
            }
        )
    )


def competing_candidates(
    candidates,
    chosen,
) -> list:
    """Return the overlap-connected redex component containing ``chosen``.

    Using the full component makes the context independent of which member of
    the same local crossroad happened to be selected first.
    """
    local_candidates = [chosen]
    local_indices = set()
    frontier_vertices = set(match_vertices(chosen[1]))

    changed = True
    while changed:
        changed = False
        for index, candidate in enumerate(candidates):
            if index in local_indices:
                continue
            candidate_vertices = set(match_vertices(candidate[1]))
            if frontier_vertices & candidate_vertices:
                local_indices.add(index)
                if candidate not in local_candidates:
                    local_candidates.append(candidate)
                new_vertices = candidate_vertices - frontier_vertices
                if new_vertices:
                    frontier_vertices.update(new_vertices)
                    changed = True

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
) -> bool:
    rewrite_obj = INSTRUMENTABLE_RULES[
        rule_name
    ]

    if isinstance(match, tuple):
        result = rewrite_obj.applier(
            g,
            match[0],
            match[1],
        )
    else:
        result = rewrite_obj.applier(
            g,
            match,
        )

    # Current PyZX appliers return bool.  Treat None as success for
    # compatibility with older versions whose appliers mutated in place but
    # did not return a value.
    return result is not False


def safe_apply_match(
    g,
    rule_name: str,
    match,
) -> bool:
    try:
        return apply_match(
            g,
            rule_name,
            match,
        )

    except Exception:
        return False


def _terminal_mu_with_greedy_continuation(
    g,
    forced_candidate=None,
) -> tuple | None:
    """Measure an extractable terminal outcome under a common continuation.

    Intermediate graph-like ZX diagrams are frequently not circuit-like, so
    trying to extract immediately after one primitive rewrite produces mostly
    missing rewards.  A forced action followed by the deterministic greedy
    policy gives every competing action the same, explicitly defined rollout.
    """
    from zx_cenf.quantale.strategies import (
        finalize_for_extraction,
        naive_greedy_reduce,
    )

    # PyZX Graph.copy() deliberately renumbers vertices.  A rewrite match
    # contains vertex IDs, so it must be applied to an ID-preserving clone.
    work = g.clone()
    if forced_candidate is not None:
        rule_name, match = forced_candidate
        if not safe_apply_match(work, rule_name, match):
            return None
        work.remove_isolated_vertices()

    naive_greedy_reduce(work)
    if not finalize_for_extraction(work):
        return None

    try:
        return mu_from_pyzx_graph(work).coords
    except Exception:
        return None


def _counterfactual_crossroad_rows(
    g,
    candidates,
    chosen,
    context_hashes,
    source_id: str | None = None,
) -> list[RewriteLogRow]:
    """Evaluate the application policy's chosen candidate for each rule."""
    from functools import cmp_to_key

    from zx_cenf.quantale.strategies import (
        STEP_PRIORITY,
        _lexicographic_better,
        _one_step_surrogate_delta,
    )

    local_candidates = competing_candidates(candidates, chosen)
    competing = tuple(sorted({rule for rule, _match in local_candidates}))
    baseline_mu = _terminal_mu_with_greedy_continuation(g)
    rows = []

    for rule_name in competing:
        scored_candidates = []
        for candidate in local_candidates:
            if candidate[0] != rule_name:
                continue
            delta = _one_step_surrogate_delta(g, *candidate)
            if delta is not None:
                scored_candidates.append((*candidate, delta))

        scored_candidates.sort(
            key=cmp_to_key(
                lambda a, b: _lexicographic_better(
                    a[2],
                    b[2],
                    priority=STEP_PRIORITY,
                )
            ),
            reverse=True,
        )
        selected_candidate = (
            scored_candidates[0][:2]
            if scored_candidates
            else None
        )
        terminal_mu = (
            _terminal_mu_with_greedy_continuation(
                g,
                forced_candidate=selected_candidate,
            )
            if selected_candidate is not None
            else None
        )

        rows.append(
            RewriteLogRow(
                pattern_hash=context_hashes[DEFAULT_CONTEXT_REPRESENTATION],
                rule=rule_name,
                was_crossroad=True,
                competing_rules=competing,
                mu_before=baseline_mu,
                mu_after=terminal_mu,
                mu_valid=baseline_mu is not None and terminal_mu is not None,
                context_hash_variants=context_hashes,
                selected_on_exploration_path=(rule_name == chosen[0]),
                source_id=source_id,
            )
        )

    return rows


def run_instrumented_reduction(
    g,
    seed: int,
    max_steps: int = 200,
    compute_mu_per_step: bool = True,
    verbose: bool = False,
    source_id: str | None = None,
) -> tuple:
    rng = random.Random(seed)

    simp.to_gh(g)

    log_rows: list[RewriteLogRow] = []

    seen_states = set()
    n_steps = 0
    stop_reason = TrainingStopReason.HIT_STEP_CAP

    for _step in range(max_steps):
        state_hash = compute_graph_state_hash(g)
        if state_hash in seen_states:
            stop_reason = TrainingStopReason.CYCLE_DETECTED
            break
        seen_states.add(state_hash)

        candidates = enumerate_all_candidates(g)

        if not candidates:
            if not simp.gadget_simp(g):
                stop_reason = TrainingStopReason.STRUCTURAL_FIXED_POINT
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

            context_hashes = compute_context_hash_variants(
                g,
                context_vertices,
            )

            pattern_hash = context_hashes[DEFAULT_CONTEXT_REPRESENTATION]

            if was_crossroad:
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

            if was_crossroad and compute_mu_per_step:
                crossroad_rows = _counterfactual_crossroad_rows(
                    g,
                    candidates,
                    chosen,
                    context_hashes,
                    source_id=source_id,
                )
            else:
                crossroad_rows = []

            applied_ok = safe_apply_match(
                g,
                rule_name,
                match,
            )

        if not applied_ok:
            stop_reason = TrainingStopReason.ALL_CANDIDATES_FAILED
            break

        g.remove_isolated_vertices()
        n_steps += 1

        if crossroad_rows:
            log_rows.extend(crossroad_rows)
        else:
            log_rows.append(
                RewriteLogRow(
                    pattern_hash=pattern_hash,
                    rule=rule_name,
                    was_crossroad=was_crossroad,
                    competing_rules=competing,
                    mu_before=None,
                    mu_after=None,
                    mu_valid=False,
                    context_hash_variants=context_hashes,
                    source_id=source_id,
                )
            )

    return (
        log_rows,
        TrainingOutcome(stop_reason, n_steps),
    )
