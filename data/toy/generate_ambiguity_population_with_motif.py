"""Generate three controlled QASM circuits with a shared structural motif."""

from pathlib import Path

import pyzx as zx


OUTPUT_DIR = Path(__file__).parent


def add_shared_motif(circ: zx.Circuit) -> None:
    """Add the identical 3-qubit motif to every circuit."""

    # Prepare the three motif qubits.
    circ.add_gate("HAD", 0)
    circ.add_gate("HAD", 1)
    circ.add_gate("HAD", 2)

    # Create the same triangle of entangling edges.
    circ.add_gate("CZ", 0, 1)
    circ.add_gate("CZ", 1, 2)
    circ.add_gate("CZ", 0, 2)


def add_small_background(circ: zx.Circuit) -> None:
    """Add a small single-qubit background."""

    gates = [
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
    ]

    for gate, qubit in gates:
        circ.add_gate(gate, qubit)


def add_medium_background(circ: zx.Circuit) -> None:
    """Add a different background for the 4-qubit circuit."""

    # Independent activity on qubit 3.
    background_q3 = [
        ("HAD", 3),
        ("T", 3),
        ("S", 3),
        ("T", 3),
        ("HAD", 3),
        ("T", 3),
        ("S", 3),
        ("T", 3),
        ("HAD", 3),
        ("T", 3),
    ]

    for gate, qubit in background_q3:
        circ.add_gate(gate, qubit)

    # Additional single-qubit structure.
    extra = [
        ("T", 0),
        ("S", 1),
        ("T", 2),
        ("HAD", 3),
        ("T", 3),
        ("S", 3),
        ("HAD", 3),
        ("T", 3),
    ]

    for gate, qubit in extra:
        circ.add_gate(gate, qubit)


def add_large_background(circ: zx.Circuit) -> None:
    """Add a different background for the 5-qubit circuit."""

    # Independent structure on qubits 3 and 4.
    background = [
        ("HAD", 3),
        ("HAD", 4),
        ("T", 3),
        ("S", 4),
        ("T", 4),
        ("HAD", 3),
        ("T", 3),
        ("HAD", 4),
        ("S", 3),
        ("T", 4),
        ("HAD", 3),
        ("T", 4),
        ("S", 4),
        ("HAD", 4),
        ("T", 3),
        ("S", 3),
        ("HAD", 3),
        ("T", 4),
    ]

    for gate, qubit in background:
        circ.add_gate(gate, qubit)

    # Additional local phase structure on the motif qubits.
    motif_local = [
        ("T", 0),
        ("S", 1),
        ("T", 2),
        ("HAD", 0),
        ("T", 1),
        ("S", 2),
        ("T", 0),
        ("HAD", 2),
        ("T", 1),
        ("S", 0),
    ]

    for gate, qubit in motif_local:
        circ.add_gate(gate, qubit)


def build_small() -> zx.Circuit:
    """Build the 3-qubit circuit."""

    circ = zx.Circuit(3)

    add_small_background(circ)
    add_shared_motif(circ)

    return circ


def build_medium() -> zx.Circuit:
    """Build the 4-qubit circuit."""

    circ = zx.Circuit(4)

    add_medium_background(circ)
    add_shared_motif(circ)

    return circ


def build_large() -> zx.Circuit:
    """Build the 5-qubit circuit."""

    circ = zx.Circuit(5)

    add_large_background(circ)
    add_shared_motif(circ)

    return circ


if __name__ == "__main__":
    circuits = [
        ("toy_small_dense", build_small()),
        ("toy_medium_diverse", build_medium()),
        ("toy_larger_clifford_heavy", build_large()),
    ]

    for name, circ in circuits:
        path = OUTPUT_DIR / f"{name}.qasm"
        path.write_text(circ.to_qasm())

        print(
            f"{path} — "
            f"{len(circ.gates)} gates, "
            f"{circ.qubits} qubits"
        )