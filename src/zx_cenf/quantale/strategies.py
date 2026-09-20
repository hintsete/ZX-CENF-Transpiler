"""Application-time strategies: naive greedy and table-guided."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import cmp_to_key

import pyzx.simplify as simp

from zx_cenf.quantale.instrumented_reduction import (
    DEFAULT_CONTEXT_REPRESENTATION,
    competing_candidates,
    compute_context_hash_variants,
    decision_context_vertices,
    enumerate_all_candidates,
    local_competing_rules,
    match_vertices,
    safe_apply_match,
)
from zx_cenf.quantale.pattern_hash import (
    compute_graph_state_hash,
    is_valid_pattern,
    make_context_key,
)
from zx_cenf.quantale.surrogate import SURROGATE_PRIORITY, surrogate_value


PRIORITY = (0, 2, 1)  # twoqubitcount, depth, tcount -- mu is 3D
STEP_PRIORITY = SURROGATE_PRIORITY


@dataclass
class TableHitRecord:
    step: int
    pattern_hash: str
    preferred_rule: str
    competing_rules: tuple
    rule_selected: str
    surrogate_before: tuple
    surrogate_after: tuple | None
    differs_from_greedy_choice: bool


class StopReason(str, Enum):
    STRUCTURAL_FIXED_POINT = "structural_fixed_point"
    NO_IMPROVING_MOVE = "no_improving_move"
    HIT_STEP_CAP = "hit_step_cap"
    ALL_CANDIDATES_FAILED = "all_candidates_failed"
    CYCLE_DETECTED = "cycle_detected"


@dataclass
class ReductionOutcome:
    stop_reason: StopReason
    n_steps: int
    n_table_hits: int
    n_table_misses: int
    lookup_diagnostics: dict = field(default_factory=dict)

    @property
    def reached_fixed_point(self) -> bool:
        return self.stop_reason == StopReason.STRUCTURAL_FIXED_POINT

    @property
    def reached_policy_fixed_point(self) -> bool:
        return self.stop_reason in (
            StopReason.STRUCTURAL_FIXED_POINT,
            StopReason.NO_IMPROVING_MOVE,
        )


def _lexicographic_better(
    delta_a,
    delta_b,
    tol: float = 1e-9,
    priority=None,
) -> int:
    """
    Compare improvement vectors.

    Deltas are defined as:

        improvement = before_cost - after_cost

    Therefore larger positive values are better because they represent
    larger reductions in circuit cost.
    """
    for coord in (PRIORITY if priority is None else priority):
        if delta_a[coord] > delta_b[coord] + tol:
            return 1
        if delta_b[coord] > delta_a[coord] + tol:
            return -1

    return 0


def is_improving(
    delta,
    tol: float = 1e-9,
    priority=None,
) -> bool:
    """
    Return True when the first significant coordinate in the priority
    ordering improves.

    delta = before_cost - after_cost, so:
        positive -> improvement
        negative -> degradation
        zero     -> no change
    """
    for coord in (PRIORITY if priority is None else priority):
        if delta[coord] > tol:
            return True

        if delta[coord] < -tol:
            return False

    return False


def _one_step_surrogate_delta(
    g,
    rule_name: str,
    match,
):
    """
    Compute surrogate improvement as:

        surrogate_before - surrogate_after

    so positive values mean improvement.
    """
    before = surrogate_value(g).coords

    # PyZX Graph.copy() renumbers vertices; rewrite matches refer to the
    # original IDs and therefore require an ID-preserving clone.
    g_copy = g.clone()

    if not safe_apply_match(
        g_copy,
        rule_name,
        match,
    ):
        return None

    g_copy.remove_isolated_vertices()

    after = surrogate_value(g_copy).coords

    return tuple(
        b - a
        for b, a in zip(before, after)
    )


def _rank_improving_candidates(
    g,
    candidates,
) -> list:
    scored = []

    for rule_name, match in candidates:
        delta = _one_step_surrogate_delta(
            g,
            rule_name,
            match,
        )

        if delta is None:
            continue

        if not is_improving(
            delta,
            priority=STEP_PRIORITY,
        ):
            continue

        scored.append(
            (
                rule_name,
                match,
                delta,
            )
        )

    scored.sort(
        key=cmp_to_key(
            lambda a, b: _lexicographic_better(
                a[2],
                b[2],
                priority=STEP_PRIORITY,
            )
        ),
        reverse=True,
    )

    return [
        (rule_name, match)
        for rule_name, match, _delta in scored
    ]


def finalize_for_extraction(
    g,
    max_rounds: int = 50,
) -> bool:
    for _ in range(max_rounds):
        changed = simp.spider_simp(g)
        changed = simp.id_simp(g) or changed

        if not changed:
            break

    g.remove_isolated_vertices()

    try:
        import pyzx as zx

        zx.extract_circuit(
            g.copy(),
            quiet=True,
        )

        return True

    except Exception:
        return False


def naive_greedy_reduce(
    g,
    max_steps: int = 200,
) -> ReductionOutcome:
    """
    Greedy baseline.

    This does not consult the learned pattern table.
    """
    simp.to_gh(g)

    n_steps = 0

    for _step in range(max_steps):
        candidates = enumerate_all_candidates(g)

        if not candidates:
            if not simp.gadget_simp(g):
                return ReductionOutcome(
                    StopReason.STRUCTURAL_FIXED_POINT,
                    n_steps,
                    0,
                    0,
                )

            continue

        ranked = _rank_improving_candidates(
            g,
            candidates,
        )

        if not ranked:
            return ReductionOutcome(
                StopReason.NO_IMPROVING_MOVE,
                n_steps,
                0,
                0,
            )

        if _apply_ranked_inplace(
            g,
            ranked,
        ) is None:
            return ReductionOutcome(
                StopReason.ALL_CANDIDATES_FAILED,
                n_steps,
                0,
                0,
            )

        n_steps += 1

    return ReductionOutcome(
        StopReason.HIT_STEP_CAP,
        n_steps,
        0,
        0,
    )


def _apply_ranked_inplace(
    g,
    ranked,
):
    """
    Try ranked candidates in order until one can actually be applied.
    """
    for rule_name, match in ranked:
        if safe_apply_match(
            g,
            rule_name,
            match,
        ):
            g.remove_isolated_vertices()

            return (
                rule_name,
                match,
            )

    return None


def _find_table_preference(
    g,
    candidates,
    ranked,
    pattern_table,
    diagnostics: dict | None = None,
):
    """
    Find a learned preference using the SAME decision-context definition
    used during training.

    Training hashes the shared decision context around a local crossroad.
    Therefore application-time lookup must also:

      1. identify a crossroad,
      2. compute decision_context_vertices(),
      3. hash that context,
      4. check the learned rule against the competing rules.

    The optional diagnostics dictionary records what happened during the
    actual held-out lookup path.
    """

    # A learned policy is specifically meant to escape choices that look
    # unattractive to the one-step surrogate.  Search greedy-ranked candidates
    # first, but do not make surrogate improvement a prerequisite for lookup.
    search_order = list(ranked)
    search_order.extend(
        candidate
        for candidate in candidates
        if candidate not in search_order
    )

    for candidate in search_order:
        rule_name, match = candidate

        competing_rules = local_competing_rules(
            candidates,
            candidate,
        )

        # Not a local crossroad.
        if len(competing_rules) <= 1:
            continue

        if diagnostics is not None:
            diagnostics["candidate_crossroad_checks"] += 1

        context_vertices = decision_context_vertices(
            g,
            candidates,
            candidate,
        )

        context_hashes = compute_context_hash_variants(g, context_vertices)
        pattern_hash = context_hashes[DEFAULT_CONTEXT_REPRESENTATION]

        if not is_valid_pattern(pattern_hash):
            if diagnostics is not None:
                diagnostics["invalid_hashes"] += 1

            continue

        if diagnostics is not None:
            diagnostics["valid_crossroad_hashes"] += 1
            diagnostics["observed_hashes"].add(pattern_hash)

        context_key = make_context_key(pattern_hash, competing_rules)

        if diagnostics is not None:
            diagnostics["observed_context_keys"].add(context_key)

        preferred_rule = pattern_table.get(context_key)

        # CASE B:
        # The test pattern is structurally represented, but there is no
        # learned preference for it.
        if preferred_rule is None:
            if diagnostics is not None:
                diagnostics["hash_not_in_table"] += 1

            continue

        if diagnostics is not None:
            diagnostics["hash_in_table"] += 1
            diagnostics["table_matches"].add(context_key)

        # CASE C:
        # A learned pattern exists, but the rule learned for it is not
        # actually competing at this test-time crossroad.
        if preferred_rule not in competing_rules:
            if diagnostics is not None:
                diagnostics["preferred_rule_not_competing"] += 1
                diagnostics["inapplicable_preferences"].append(
                    (
                        pattern_hash,
                        preferred_rule,
                        competing_rules,
                    )
                )

            continue

        local_candidates = competing_candidates(candidates, candidate)
        preferred_options = []
        for local_candidate in local_candidates:
            if local_candidate[0] != preferred_rule:
                continue
            delta = _one_step_surrogate_delta(g, *local_candidate)
            if delta is not None:
                preferred_options.append((*local_candidate, delta))

        if not preferred_options:
            if diagnostics is not None:
                diagnostics["preferred_rule_not_ranked"] += 1
                diagnostics["inapplicable_preferences"].append(
                    (
                        pattern_hash,
                        preferred_rule,
                        competing_rules,
                    )
                )

            continue

        preferred_options.sort(
            key=cmp_to_key(
                lambda a, b: _lexicographic_better(
                    a[2],
                    b[2],
                    priority=STEP_PRIORITY,
                )
            ),
            reverse=True,
        )
        preferred_candidate = preferred_options[0][:2]

        # CASE D:
        # Actual learned preference is applicable.
        if diagnostics is not None:
            diagnostics["applicable_preferences"] += 1
            diagnostics["applicable_hits"].append(
                (
                    pattern_hash,
                    preferred_rule,
                    competing_rules,
                )
            )

        return {
            "pattern_hash": pattern_hash,
            "context_key": context_key,
            "preferred_rule": preferred_rule,
            "preferred_candidate": preferred_candidate,
            "competing_rules": competing_rules,
        }

    return None


def _new_lookup_diagnostics() -> dict:
    """Counters for the actual table-guided held-out trajectory."""
    return {
        "candidate_crossroad_checks": 0,
        "valid_crossroad_hashes": 0,
        "invalid_hashes": 0,
        "hash_not_in_table": 0,
        "hash_in_table": 0,
        "preferred_rule_not_competing": 0,
        "preferred_rule_not_ranked": 0,
        "applicable_preferences": 0,
        "observed_hashes": set(),
        "observed_context_keys": set(),
        "table_matches": set(),
        "inapplicable_preferences": [],
        "applicable_hits": [],
    }


def _print_lookup_diagnostics(
    diagnostics: dict,
) -> None:
    """
    Print an aggregate diagnostic for the actual table-guided trajectory.

    This is intentionally aggregate rather than one line per step, so the
    Phase F output remains readable.
    """
    print("\n  --- Actual table lookup diagnostic ---")

    print(
        "  crossroad candidate checks: "
        f"{diagnostics['candidate_crossroad_checks']}"
    )

    print(
        "  valid crossroad hashes:     "
        f"{diagnostics['valid_crossroad_hashes']}"
    )

    print(
        "  unique test hashes seen:    "
        f"{len(diagnostics['observed_hashes'])}"
    )

    print(
        "  unique decision contexts:   "
        f"{len(diagnostics['observed_context_keys'])}"
    )

    print(
        "  context checks in table:    "
        f"{diagnostics['hash_in_table']}"
    )

    print(
        "  unique table contexts hit:  "
        f"{len(diagnostics['table_matches'])}"
    )

    print(
        "  context absent from table:  "
        f"{diagnostics['hash_not_in_table']}"
    )

    print(
        "  preferred rule not competing: "
        f"{diagnostics['preferred_rule_not_competing']}"
    )

    print(
        "  preferred rule not ranked:    "
        f"{diagnostics['preferred_rule_not_ranked']}"
    )

    print(
        "  applicable learned preferences: "
        f"{diagnostics['applicable_preferences']}"
    )

    if diagnostics["inapplicable_preferences"]:
        print(
            "  inapplicable examples:"
        )

        for (
            pattern_hash,
            preferred_rule,
            competing_rules,
        ) in diagnostics["inapplicable_preferences"][:5]:
            print(
                "    "
                f"hash={pattern_hash} "
                f"preferred={preferred_rule} "
                f"competing={competing_rules}"
            )

    if diagnostics["applicable_hits"]:
        print(
            "  applicable examples:"
        )

        for (
            pattern_hash,
            preferred_rule,
            competing_rules,
        ) in diagnostics["applicable_hits"][:5]:
            print(
                "    "
                f"hash={pattern_hash} "
                f"preferred={preferred_rule} "
                f"competing={competing_rules}"
            )

    print("  --------------------------------------")


def table_guided_reduce(
    g,
    pattern_table: dict,
    max_steps: int = 200,
    diagnostic: bool = False,
) -> tuple:
    simp.to_gh(g)

    n_steps = 0
    n_table_hits = 0
    n_table_misses = 0
    trace: list = []
    diagnostics = _new_lookup_diagnostics()
    seen_states = set()

    def finish(stop_reason):
        outcome = ReductionOutcome(
            stop_reason,
            n_steps,
            n_table_hits,
            n_table_misses,
            lookup_diagnostics=diagnostics,
        )
        if diagnostic:
            _print_lookup_diagnostics(diagnostics)
        return outcome, trace

    for _step in range(max_steps):
        state_hash = compute_graph_state_hash(g)
        if state_hash in seen_states:
            return finish(StopReason.CYCLE_DETECTED)
        seen_states.add(state_hash)

        candidates = enumerate_all_candidates(g)

        if not candidates:
            if not simp.gadget_simp(g):
                return finish(StopReason.STRUCTURAL_FIXED_POINT)

            continue

        ranked = _rank_improving_candidates(
            g,
            candidates,
        )

        greedy_choice_rule = ranked[0][0] if ranked else None
        has_crossroad = any(
            len(local_competing_rules(candidates, candidate)) > 1
            for candidate in candidates
        )

        lookup = _find_table_preference(
            g,
            candidates,
            ranked,
            pattern_table,
            diagnostics=diagnostics,
        )

        if lookup is not None:
            preferred_candidate = lookup["preferred_candidate"]
            surrogate_before = surrogate_value(g).coords
            ordered = [
                preferred_candidate
            ] + [
                candidate
                for candidate in ranked
                if candidate != preferred_candidate
            ]

        else:
            if has_crossroad:
                n_table_misses += 1
            if not ranked:
                return finish(StopReason.NO_IMPROVING_MOVE)
            ordered = ranked

        applied = _apply_ranked_inplace(
            g,
            ordered,
        )

        if applied is None:
            return finish(StopReason.ALL_CANDIDATES_FAILED)

        n_steps += 1

        if lookup is not None and applied == lookup["preferred_candidate"]:
            n_table_hits += 1
            applied_rule, _match = applied
            trace.append(
                TableHitRecord(
                    step=_step,
                    pattern_hash=lookup["pattern_hash"],
                    preferred_rule=lookup["preferred_rule"],
                    competing_rules=lookup["competing_rules"],
                    rule_selected=applied_rule,
                    surrogate_before=surrogate_before,
                    surrogate_after=surrogate_value(g).coords,
                    differs_from_greedy_choice=(
                        greedy_choice_rule is None
                        or applied_rule != greedy_choice_rule
                    ),
                )
            )
        elif lookup is not None and has_crossroad:
            n_table_misses += 1

    return finish(StopReason.HIT_STEP_CAP)