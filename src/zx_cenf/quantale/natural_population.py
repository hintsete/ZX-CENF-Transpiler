"""Deterministic, non-planted circuit population for the Phase F study."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from difflib import SequenceMatcher

import pyzx as zx


@dataclass(frozen=True)
class NaturalCircuitSpec:
    circuit_id: str
    split: str
    family: str
    qubits: int
    depth: int
    p_had: float
    p_t: float
    seed: int


# These are standard stochastic gate-mixture regimes, not planted rewrite
# motifs.  The remaining probability mass produces CNOT gates.
NATURAL_FAMILIES = {
    "clifford_heavy": (0.30, 0.08),
    "balanced": (0.20, 0.22),
    "phase_rich": (0.12, 0.38),
    "entangling_heavy": (0.12, 0.15),
}


def make_population_specs(
    train_per_family: int = 10,
    test_per_family: int = 5,
    base_seed: int = 17_000,
) -> list[NaturalCircuitSpec]:
    """Return a stratified circuit-level split with disjoint random seeds."""
    specs = []
    for family_index, (family, (p_had, p_t)) in enumerate(NATURAL_FAMILIES.items()):
        for split, count, split_offset in (
            ("train", train_per_family, 0),
            ("test", test_per_family, 10_000),
        ):
            for index in range(count):
                # Both size coordinates vary within every family and split.
                qubits = 3 + ((index + 2 * family_index + split_offset) % 4)
                depth = 18 + 4 * ((2 * index + family_index + split_offset) % 5)
                seed = base_seed + 1_000 * family_index + split_offset + index
                specs.append(
                    NaturalCircuitSpec(
                        circuit_id=f"{split}_{family}_{index:03d}",
                        split=split,
                        family=family,
                        qubits=qubits,
                        depth=depth,
                        p_had=p_had,
                        p_t=p_t,
                        seed=seed,
                    )
                )
    return specs


def build_natural_circuit(spec: NaturalCircuitSpec) -> zx.Circuit:
    """Generate one circuit without inserting or selecting for a motif."""
    return zx.generate.CNOT_HAD_PHASE_circuit(
        spec.qubits,
        spec.depth,
        p_had=spec.p_had,
        p_t=spec.p_t,
        seed=spec.seed,
    )


def circuit_tokens(circuit: zx.Circuit) -> tuple[str, ...]:
    return tuple(str(gate) for gate in circuit.gates)


def circuit_fingerprint(circuit: zx.Circuit) -> str:
    payload = f"qubits={circuit.qubits}\n" + "\n".join(circuit_tokens(circuit))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def audit_train_test_separation(
    circuits: list[tuple[NaturalCircuitSpec, zx.Circuit]],
    near_duplicate_threshold: float = 0.85,
) -> dict:
    """Audit exact identity and order-sensitive gate-sequence similarity."""
    train = [(spec, circuit) for spec, circuit in circuits if spec.split == "train"]
    test = [(spec, circuit) for spec, circuit in circuits if spec.split == "test"]
    fingerprints = {
        spec.circuit_id: circuit_fingerprint(circuit)
        for spec, circuit in circuits
    }
    exact_pairs = []
    near_pairs = []
    max_similarity = 0.0
    max_pair = None
    for train_spec, train_circuit in train:
        for test_spec, test_circuit in test:
            if fingerprints[train_spec.circuit_id] == fingerprints[test_spec.circuit_id]:
                exact_pairs.append((train_spec.circuit_id, test_spec.circuit_id))
            similarity = SequenceMatcher(
                None,
                circuit_tokens(train_circuit),
                circuit_tokens(test_circuit),
                autojunk=False,
            ).ratio()
            if similarity > max_similarity:
                max_similarity = similarity
                max_pair = (train_spec.circuit_id, test_spec.circuit_id)
            if similarity >= near_duplicate_threshold:
                near_pairs.append(
                    (train_spec.circuit_id, test_spec.circuit_id, similarity)
                )
    return {
        "n_exact_train_test_duplicates": len(exact_pairs),
        "exact_pairs": exact_pairs,
        "near_duplicate_threshold": near_duplicate_threshold,
        "n_near_duplicate_train_test_pairs": len(near_pairs),
        "near_pairs": near_pairs,
        "max_train_test_similarity": max_similarity,
        "max_similarity_pair": max_pair,
    }
