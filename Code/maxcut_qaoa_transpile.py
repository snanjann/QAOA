"""Transpilation comparison code for the 4-qubit Max-Cut QAOA circuit only.

Package installation:
    pip install qiskit numpy qiskit-aer qiskit-ibm-runtime
or
    python -m pip install qiskit numpy qiskit-aer qiskit-ibm-runtime

Imported libraries:
    - time: runtime measurement
    - qiskit.transpile: baseline hardware mapping
    - qiskit.transpiler.preset_passmanagers.generate_preset_pass_manager:
      optimized transpilation workflow
    - qiskit_ibm_runtime.fake_provider.FakeManilaV2: realistic fake backend

This script:
    1. Reuses the best logical Max-Cut QAOA parameters from the ideal script.
    2. Builds the measured logical circuit.
    3. Compares baseline transpilation with an optimized transpilation.
    4. Prints metrics and OpenQASM for both circuits.
    5. Prints the runtime of the whole script.
"""

from __future__ import annotations

import time
from typing import Any

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeManilaV2

from maxcut_qaoa_ideal import EDGES, NUM_QUBITS, build_qaoa_circuit, find_best_qaoa_parameters, qasm_output


def two_qubit_gate_count(circuit: QuantumCircuit) -> int:
    """Count the main two-qubit gates in a transpiled circuit."""
    counts = circuit.count_ops()
    return sum(int(counts.get(name, 0)) for name in ("cx", "cz", "ecr", "swap", "rzz"))


def circuit_metrics(circuit: QuantumCircuit) -> dict[str, Any]:
    """Return a small metric set for transpilation comparison."""
    return {
        "depth": circuit.depth(),
        "size": circuit.size(),
        "two_qubit_gates": two_qubit_gate_count(circuit),
        "ops": dict(circuit.count_ops()),
    }


def print_metrics(title: str, circuit: QuantumCircuit) -> None:
    """Print depth, size, two-qubit count, and gate histogram."""
    metrics = circuit_metrics(circuit)
    print(f"=== {title} ===")
    print(f"Depth: {metrics['depth']}")
    print(f"Size: {metrics['size']}")
    print(f"Two-qubit gates: {metrics['two_qubit_gates']}")
    print(f"Operation counts: {metrics['ops']}")
    print()


def main() -> None:
    start_time = time.perf_counter()

    best_result = find_best_qaoa_parameters()
    logical_circuit = build_qaoa_circuit(best_result.gamma, best_result.beta, measure=True)
    backend = FakeManilaV2()

    # Baseline: direct transpilation with no aggressive optimization.
    baseline = transpile(logical_circuit, backend=backend, optimization_level=0)

    # Optimized: preset pass manager with aggressive optimization and SABRE.
    optimized_pass_manager = generate_preset_pass_manager(
        backend=backend,
        optimization_level=3,
        layout_method="sabre",
        routing_method="sabre",
        seed_transpiler=123,
    )
    optimized = optimized_pass_manager.run(logical_circuit)

    base_metrics = circuit_metrics(baseline)
    opt_metrics = circuit_metrics(optimized)
    elapsed = time.perf_counter() - start_time

    print("=== Package installation ===")
    print("pip install qiskit numpy qiskit-aer qiskit-ibm-runtime")
    print()

    print("=== 4-Qubit Max-Cut transpilation setup ===")
    print(f"Qubits: {NUM_QUBITS}")
    print(f"Edges: {EDGES}")
    print(f"Best gamma: {best_result.gamma:.6f}")
    print(f"Best beta:  {best_result.beta:.6f}")
    print(f"Backend: {backend.name}")
    print()

    print_metrics("Logical circuit before transpilation", logical_circuit)
    print_metrics("Baseline transpilation", baseline)
    print_metrics("Optimized transpilation", optimized)

    print("=== Comparison summary ===")
    print(f"Depth: {base_metrics['depth']} -> {opt_metrics['depth']}")
    print(f"Size: {base_metrics['size']} -> {opt_metrics['size']}")
    print(f"Two-qubit gates: {base_metrics['two_qubit_gates']} -> {opt_metrics['two_qubit_gates']}")
    print()

    print("=== Baseline transpiled circuit (OpenQASM) ===")
    print(qasm_output(baseline))
    print()
    print("=== Optimized transpiled circuit (OpenQASM) ===")
    print(qasm_output(optimized))
    print()
    print(f"Runtime (seconds): {elapsed:.6f}")


if __name__ == "__main__":
    main()
