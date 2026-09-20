"""End-to-end invariants for the Phase F training/application contract."""

from __future__ import annotations

from collections import defaultdict

import pyzx as zx

from zx_cenf.quantale.instrumented_reduction import (
    RewriteLogRow,
    run_instrumented_reduction,
)
from zx_cenf.quantale.connected_motifs import (
    MOTIF_SPECS,
    build_connected_motif_graph,
    collect_target_action_observations as collect_connected_observations,
    find_target_crossroad,
)
from zx_cenf.quantale.pattern_hash import make_context_key
from zx_cenf.quantale.pattern_table import build_pattern_rule_table
from zx_cenf.quantale.natural_population import (
    audit_train_test_separation,
    build_natural_circuit,
    make_population_specs,
)
from zx_cenf.quantale.null_table import build_wrong_rule_table
from zx_cenf.quantale.positive_control import (
    EXPECTED_PREFERRED_RULE,
    INTENTIONALLY_WRONG_RULE,
    TARGET_CONTEXT_KEY,
    build_positive_control_graph,
    collect_target_action_observations,
)
from zx_cenf.quantale.strategies import (
    finalize_for_extraction,
    naive_greedy_reduce,
    table_guided_reduce,
)
from zx_cenf.quantale.valuation import mu_from_pyzx_graph


def _small_motif_graph():
    circuit = zx.Circuit(3)
    for gate, qubit in (
        ("T", 0),
        ("S", 1),
        ("T", 2),
        ("HAD", 0),
        ("T", 1),
        ("S", 2),
        ("T", 0),
        ("HAD", 2),
        ("S", 1),
        ("T", 2),
        ("HAD", 0),
        ("HAD", 1),
        ("HAD", 2),
    ):
        circuit.add_gate(gate, qubit)
    circuit.add_gate("CZ", 0, 1)
    circuit.add_gate("CZ", 1, 2)
    circuit.add_gate("CZ", 0, 2)
    return circuit.to_graph()


def test_context_key_includes_the_available_rule_set():
    assert make_context_key("hash", ("pivot", "lcomp")) == (
        "hash",
        ("lcomp", "pivot"),
    )
    assert make_context_key("hash", ("pivot",)) != make_context_key(
        "hash",
        ("pivot", "lcomp"),
    )


def test_pattern_table_learns_from_balanced_terminal_rollouts():
    rows = []
    for _ in range(5):
        rows.extend(
            [
                RewriteLogRow(
                    pattern_hash="shared",
                    rule="pivot",
                    was_crossroad=True,
                    competing_rules=("pivot", "lcomp"),
                    mu_before=(10.0, 5.0, 20.0),
                    mu_after=(8.0, 5.0, 18.0),
                ),
                RewriteLogRow(
                    pattern_hash="shared",
                    rule="lcomp",
                    was_crossroad=True,
                    competing_rules=("pivot", "lcomp"),
                    mu_before=(10.0, 5.0, 20.0),
                    mu_after=(9.0, 4.0, 17.0),
                ),
            ]
        )

    table, diagnostics = build_pattern_rule_table(rows, min_occurrences=5)

    assert table[make_context_key("shared", ("pivot", "lcomp"))] == "pivot"
    assert diagnostics["n_patterns_one_sided"] == 0
    assert diagnostics["n_patterns_learned"] == 1


def test_pattern_table_does_not_ignore_an_unsupported_competing_rule():
    rows = []
    for _ in range(5):
        for rule, after in (
            ("pivot", (8.0, 5.0, 18.0)),
            ("lcomp", (9.0, 4.0, 17.0)),
        ):
            rows.append(
                RewriteLogRow(
                    pattern_hash="three-way",
                    rule=rule,
                    was_crossroad=True,
                    competing_rules=("pivot", "lcomp", "pivot_gadget"),
                    mu_before=(10.0, 5.0, 20.0),
                    mu_after=after,
                )
            )

    table, diagnostics = build_pattern_rule_table(rows, min_occurrences=5)

    assert table == {}
    assert diagnostics["n_patterns_incomplete"] == 1


def test_pattern_table_can_require_evidence_from_distinct_circuits():
    rows = []
    for source_id in ("circuit-a", "circuit-b"):
        for _ in range(3):
            for rule, after in (("pivot", (8, 5, 18)), ("lcomp", (9, 4, 17))):
                rows.append(
                    RewriteLogRow(
                        pattern_hash="shared",
                        rule=rule,
                        was_crossroad=True,
                        competing_rules=("pivot", "lcomp"),
                        mu_before=(10, 5, 20),
                        mu_after=after,
                        source_id=source_id,
                    )
                )

    table, diagnostics = build_pattern_rule_table(
        rows,
        min_occurrences=5,
        min_distinct_sources=3,
    )

    assert table == {}
    assert diagnostics["n_patterns_adequately_supported"] == 0
    assert diagnostics["n_patterns_insufficient_evidence"] == 1


def test_wrong_rule_table_chooses_a_supported_alternative():
    rows = []
    for source_index in range(5):
        for rule, after in (("pivot", (8, 5, 18)), ("lcomp", (9, 4, 17))):
            rows.append(
                RewriteLogRow(
                    pattern_hash="shared",
                    rule=rule,
                    was_crossroad=True,
                    competing_rules=("pivot", "lcomp"),
                    mu_before=(10, 5, 20),
                    mu_after=after,
                    source_id=f"circuit-{source_index}",
                )
            )
    table, diagnostics = build_pattern_rule_table(
        rows, min_occurrences=5, min_distinct_sources=5
    )
    key = make_context_key("shared", ("pivot", "lcomp"))

    assert table[key] == "pivot"
    assert build_wrong_rule_table(table, diagnostics)[key] == "lcomp"


def test_natural_population_has_a_clean_reproducible_circuit_split():
    specs = make_population_specs(train_per_family=2, test_per_family=1)
    circuits = [(spec, build_natural_circuit(spec)) for spec in specs]
    audit = audit_train_test_separation(circuits)

    assert len({spec.circuit_id for spec in specs}) == len(specs)
    assert audit["n_exact_train_test_duplicates"] == 0
    assert audit["n_near_duplicate_train_test_pairs"] == 0


def test_natural_population_seed_changes_circuits_without_changing_protocol():
    first = make_population_specs(train_per_family=1, test_per_family=1, base_seed=100)
    second = make_population_specs(train_per_family=1, test_per_family=1, base_seed=200)

    assert [spec.circuit_id for spec in first] == [spec.circuit_id for spec in second]
    assert [(spec.family, spec.qubits, spec.depth) for spec in first] == [
        (spec.family, spec.qubits, spec.depth) for spec in second
    ]
    assert [spec.seed for spec in first] != [spec.seed for spec in second]


def test_training_records_each_competing_rule_with_terminal_mu():
    rows, outcome = run_instrumented_reduction(_small_motif_graph(), seed=0)
    crossroad_rows = [row for row in rows if row.was_crossroad]
    assert crossroad_rows
    assert all(row.mu_valid for row in crossroad_rows)

    observed = defaultdict(set)
    expected = {}
    for row in crossroad_rows:
        key = make_context_key(row.pattern_hash, row.competing_rules)
        observed[key].add(row.rule)
        expected[key] = set(row.competing_rules)

    assert all(observed[key] == rules for key, rules in expected.items())
    assert outcome.n_steps > 0


def test_table_lookup_uses_contexts_from_the_actual_application_trajectory():
    first_graph = _small_motif_graph()
    empty_outcome, _ = table_guided_reduce(first_graph, {})
    observed_keys = empty_outcome.lookup_diagnostics["observed_context_keys"]
    assert observed_keys

    pattern_table = {
        context_key: context_key[1][0]
        for context_key in observed_keys
    }
    second_graph = _small_motif_graph()
    guided_outcome, trace = table_guided_reduce(
        second_graph,
        pattern_table,
    )

    assert guided_outcome.n_table_hits >= 1
    assert trace


def test_pyzx_clone_preserves_sparse_vertex_ids_but_copy_does_not():
    graph = zx.Graph()
    vertices = [graph.add_vertex(zx.VertexType.Z) for _ in range(4)]
    graph.remove_vertex(vertices[1])

    assert set(graph.clone().vertices()) == set(graph.vertices())
    assert set(graph.copy().vertices()) != set(graph.vertices())


def _terminal_mu(graph, table=None):
    work = graph.clone()
    if table is None:
        naive_greedy_reduce(work)
        trace = []
    else:
        _outcome, trace = table_guided_reduce(work, table)
    assert finalize_for_extraction(work)
    return mu_from_pyzx_graph(work).coords, trace


def test_positive_control_learns_and_transfers_expected_rule():
    rows = []
    for index in range(3):
        rows.extend(
            collect_target_action_observations(
                build_positive_control_graph(index, "train")
            )
        )

    table, _diagnostics = build_pattern_rule_table(rows, min_occurrences=3)
    assert table[TARGET_CONTEXT_KEY] == EXPECTED_PREFERRED_RULE

    held_out = build_positive_control_graph(0, "test")
    greedy_mu, _ = _terminal_mu(held_out)
    learned_mu, learned_trace = _terminal_mu(held_out, table)
    wrong_mu, _ = _terminal_mu(
        held_out,
        {TARGET_CONTEXT_KEY: INTENTIONALLY_WRONG_RULE},
    )

    assert any(record.differs_from_greedy_choice for record in learned_trace)
    assert learned_mu[0] == greedy_mu[0] - 3
    assert learned_mu[2] == greedy_mu[2] + 2
    assert wrong_mu == greedy_mu


def test_connected_motif_population_preserves_distinct_decision_contexts():
    rows = []
    expected_keys = set()
    for spec in MOTIF_SPECS:
        graph = build_connected_motif_graph(spec, 0, "train")
        find_target_crossroad(graph, spec)
        observations = collect_connected_observations(graph, spec)
        assert {row.rule for row in observations} == set(spec.competing_rules)
        assert all(row.mu_valid for row in observations)
        rows.extend(observations)
        expected_keys.add(spec.context_key)

    table, diagnostics = build_pattern_rule_table(rows, min_occurrences=1)

    assert set(table) == expected_keys
    assert diagnostics["n_patterns_incomplete"] == 0

    # Regression: learned guidance must be able to leave a one-step surrogate
    # local minimum.  This family was structurally present but unreachable
    # when table lookup was incorrectly restricted to improving candidates.
    spec = MOTIF_SPECS[0]
    held_out = build_connected_motif_graph(spec, 0, "test")
    outcome, trace = table_guided_reduce(held_out, table)
    assert outcome.n_table_hits >= 1
    assert any(record.pattern_hash == spec.pattern_hash for record in trace)
