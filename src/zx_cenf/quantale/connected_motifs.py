"""Connected multi-motif population for the Phase F transfer experiment.

Each family is a reproducible, circuit-derived non-confluent crossroad.  A
random circuit is tensor-composed with the motif and then connected to it by a
Hadamard edge.  The attachment is chosen outside the radius-one decision
context, so the full diagram is connected without changing the context that
is supposed to recur between training and test populations.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import networkx as nx
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import EdgeType

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


@dataclass(frozen=True)
class MotifSpec:
    name: str
    circuit_seed: int
    rewrite_prefix_steps: int
    pattern_hash: str
    competing_rules: tuple[str, ...]
    expected_preferred_rule: str
    intentionally_wrong_rule: str

    @property
    def context_key(self) -> tuple[str, tuple[str, ...]]:
        return make_context_key(self.pattern_hash, self.competing_rules)


# These states were found by deterministic exploration of independently
# generated three-qubit circuits.  In particular, the first two families have
# the same action set but opposite terminal preferences.
MOTIF_SPECS = (
    MotifSpec(
        name="boundary_over_gadget",
        circuit_seed=8,
        rewrite_prefix_steps=25,
        pattern_hash="a3779bb05773e75b2e3bd5c4b499dc2e",
        competing_rules=("pivot_boundary", "pivot_gadget"),
        expected_preferred_rule="pivot_boundary",
        intentionally_wrong_rule="pivot_gadget",
    ),
    MotifSpec(
        name="gadget_over_boundary",
        circuit_seed=9,
        rewrite_prefix_steps=21,
        pattern_hash="3e99fb0c9dcce563276bba4bc9809815",
        competing_rules=("pivot_boundary", "pivot_gadget"),
        expected_preferred_rule="pivot_gadget",
        intentionally_wrong_rule="pivot_boundary",
    ),
    MotifSpec(
        name="gadget_over_pivot",
        circuit_seed=10,
        rewrite_prefix_steps=11,
        pattern_hash="7df5c8ce565cfcc266b52ae025ca9de7",
        competing_rules=("pivot", "pivot_gadget"),
        expected_preferred_rule="pivot_gadget",
        intentionally_wrong_rule="pivot",
    ),
    MotifSpec(
        name="boundary_three_way",
        circuit_seed=11,
        rewrite_prefix_steps=21,
        pattern_hash="0ddb8baeb05f45e15cad9d55a164f9d4",
        competing_rules=("lcomp", "pivot_boundary", "pivot_gadget"),
        expected_preferred_rule="pivot_boundary",
        intentionally_wrong_rule="pivot_gadget",
    ),
    MotifSpec(
        name="gadget_four_way",
        circuit_seed=12,
        rewrite_prefix_steps=9,
        pattern_hash="b6c38a00b099b610df764ef3e5f874fd",
        competing_rules=(
            "id_removal",
            "pivot",
            "pivot_boundary",
            "pivot_gadget",
        ),
        expected_preferred_rule="pivot_gadget",
        intentionally_wrong_rule="pivot",
    ),
    MotifSpec(
        name="gadget_three_way",
        circuit_seed=18,
        rewrite_prefix_steps=20,
        pattern_hash="5d9f111d2fc501e550d80b9957250932",
        competing_rules=("pivot", "pivot_boundary", "pivot_gadget"),
        expected_preferred_rule="pivot_gadget",
        intentionally_wrong_rule="pivot_boundary",
    ),
)


def build_motif_graph(spec: MotifSpec):
    """Replay exploration up to a family's designated decision state."""
    circuit = zx.generate.CNOT_HAD_PHASE_circuit(
        3,
        20,
        p_had=0.25,
        p_t=0.35,
        clifford=False,
        seed=spec.circuit_seed,
    )
    graph = circuit.to_graph()
    simp.to_gh(graph)
    rng = random.Random(0)

    n_applied = 0
    while n_applied < spec.rewrite_prefix_steps:
        candidates = enumerate_all_candidates(graph)
        if not candidates:
            if not simp.gadget_simp(graph):
                raise RuntimeError(
                    f"{spec.name} reached a fixed point before its target"
                )
            continue

        remaining = list(candidates)
        rng.shuffle(remaining)
        applied = False
        while remaining and not applied:
            applied = safe_apply_match(graph, *remaining.pop())

        if not applied:
            raise RuntimeError(f"could not replay motif {spec.name}")

        graph.remove_isolated_vertices()
        n_applied += 1

    find_target_crossroad(graph, spec)
    return graph


def build_connected_motif_graph(spec: MotifSpec, index: int, split: str):
    """Embed one motif in a held-out connected randomized background."""
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    if index < 0:
        raise ValueError("index must be non-negative")

    motif = build_motif_graph(spec)
    family_index = MOTIF_SPECS.index(spec)
    seed_base = 10_000 if split == "train" else 1_000_000
    seed = seed_base + 10_000 * family_index + 101 * index
    background = _connected_background(index, seed)

    motif_output_count = len(motif.outputs())
    graph = motif.tensor(background)
    target = find_target_crossroad(graph, spec)
    context_vertices = target[3]

    outputs = graph.outputs()
    motif_outputs = outputs[:motif_output_count]
    background_outputs = outputs[motif_output_count:]
    motif_attachment = _farthest_output_neighbour(
        graph,
        motif_outputs,
        context_vertices,
    )
    background_attachment = next(
        iter(graph.neighbors(background_outputs[0]))
    )
    graph.add_edge(
        (motif_attachment, background_attachment),
        EdgeType.HADAMARD,
    )

    if not _is_connected(graph):
        raise RuntimeError(f"connected embedding failed for {spec.name}")

    # This also proves that the attachment did not enter the hashed ego graph
    # or otherwise alter the intended action set.
    find_target_crossroad(graph, spec)
    return graph


def collect_target_action_observations(graph, spec: MotifSpec) -> list:
    """Evaluate every action at one family's designated crossroad."""
    work = graph.clone()
    simp.to_gh(work)
    candidates, chosen, context_hashes, _vertices = find_target_crossroad(
        work,
        spec,
    )
    rows = _counterfactual_crossroad_rows(
        work,
        candidates,
        chosen,
        context_hashes,
    )

    observed_rules = {row.rule for row in rows}
    if observed_rules != set(spec.competing_rules):
        raise RuntimeError(
            f"{spec.name} action set changed: expected "
            f"{spec.competing_rules}, got {sorted(observed_rules)}"
        )
    if not all(row.mu_valid for row in rows):
        raise RuntimeError(f"terminal rollout failed for {spec.name}")
    return rows


def find_target_crossroad(graph, spec: MotifSpec):
    """Return the candidate component matching ``spec.context_key``."""
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
        if context_key == spec.context_key:
            return candidates, candidate, context_hashes, context_vertices

    raise RuntimeError(
        f"target crossroad for {spec.name} was not found; "
        f"observed {sorted(observed_keys)!r}"
    )


def _connected_background(index: int, seed: int):
    """Generate a connected circuit graph, retrying deterministic seeds."""
    for attempt in range(100):
        graph = zx.generate.CNOT_HAD_PHASE_circuit(
            2 + index % 3,
            5 + index % 8,
            p_had=0.30,
            p_t=0.40,
            clifford=False,
            seed=seed + attempt,
        ).to_graph()
        simp.to_gh(graph)
        if _is_connected(graph):
            return graph
    raise RuntimeError("could not generate a connected random background")


def _as_networkx(graph) -> nx.Graph:
    result = nx.Graph()
    result.add_nodes_from(graph.vertices())
    result.add_edges_from(graph.edge_st(edge) for edge in graph.edges())
    return result


def _is_connected(graph) -> bool:
    nx_graph = _as_networkx(graph)
    return bool(nx_graph) and nx.is_connected(nx_graph)


def _farthest_output_neighbour(graph, outputs, context_vertices) -> int:
    nx_graph = _as_networkx(graph)
    choices = []
    for output in outputs:
        neighbour = next(iter(graph.neighbors(output)))
        distance = min(
            nx.shortest_path_length(nx_graph, center, neighbour)
            for center in context_vertices
        )
        choices.append((distance, neighbour))

    distance, neighbour = max(choices)
    # At distance one the attachment spider is already in the radius-one ego
    # graph, but the new neighbour is at distance two.  NO_DEGREE deliberately
    # omits the attachment spider's global degree, so this still leaves the
    # active representation unchanged.  Distance zero would add a new node to
    # the hashed ego graph and is therefore rejected.
    if distance < 1:
        raise RuntimeError(
            "no safe output attachment exists outside the context centres"
        )
    return neighbour