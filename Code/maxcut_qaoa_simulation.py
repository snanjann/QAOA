"""Simulation code for the 4-qubit Max-Cut QAOA circuit only.

Package installation:
    pip install qiskit numpy qiskit-aer qiskit-ibm-runtime
or
    python -m pip install qiskit numpy qiskit-aer qiskit-ibm-runtime

Imported libraries:
    - time: runtime measurement
    - statistics.mean: average sampled cut value
    - qiskit.transpile: simulator-ready compilation
    - qiskit_aer.AerSimulator: ideal and noisy simulation
    - qiskit_ibm_runtime.fake_provider.FakeManilaV2: noise model source
    - qiskit.transpiler.preset_passmanagers.generate_preset_pass_manager:
      optimized physical compilation before noisy simulation

This script:
    1. Reuses the best Max-Cut QAOA parameters from the ideal script.
    2. Runs ideal shot-based simulation.
    3. Runs backend-like noisy simulation.
    4. Converts sampled bitstrings to Max-Cut values.
    5. Prints both result summaries and total runtime.
"""

from __future__ import annotations

import time
from statistics import mean
from typing import Any

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from maxcut_qaoa_ideal import EDGES, NUM_QUBITS, build_qaoa_circuit, find_best_qaoa_parameters, maxcut_cost, qasm_output


def summarize_counts(counts: dict[str, int]) -> dict[str, Any]:
    """Convert sampled counts into Max-Cut statistics."""
    expanded_cut_values: list[int] = []
    best_cut = -1
    best_bitstrings: list[str] = []

    for bitstring, count in counts.items():
        cut_value = maxcut_cost(bitstring, EDGES)
        expanded_cut_values.extend([cut_value] * count)
        if cut_value > best_cut:
            best_cut = cut_value
            best_bitstrings = [bitstring]
        elif cut_value == best_cut:
            best_bitstrings.append(bitstring)

    most_frequent = max(counts, key=counts.get)
    return {
        "most_frequent_bitstring": most_frequent,
        "most_frequent_count": counts[most_frequent],
        "most_frequent_cut": maxcut_cost(most_frequent, EDGES),
        "best_sampled_cut": best_cut,
        "best_sampled_bitstrings": best_bitstrings,
        "average_sampled_cut": mean(expanded_cut_values) if expanded_cut_values else 0.0,
        "unique_bitstrings": len(counts),
    }


def print_summary(title: str, counts: dict[str, int]) -> None:
    """Print the simulation summary for a counts dictionary."""
    summary = summarize_counts(counts)
    print(f"=== {title} ===")
    print(f"Counts: {counts}")
    print(f"Unique bitstrings: {summary['unique_bitstrings']}")
    print(f"Most frequent bitstring: {summary['most_frequent_bitstring']}")
    print(f"Its count: {summary['most_frequent_count']}")
    print(f"Its cut value: {summary['most_frequent_cut']}")
    print(f"Best sampled cut: {summary['best_sampled_cut']}")
    print(f"Best sampled bitstrings: {summary['best_sampled_bitstrings']}")
    print(f"Average sampled cut: {summary['average_sampled_cut']:.4f}")
    print()


def main() -> None:
    start_time = time.perf_counter()

    shots = 4096
    best_result = find_best_qaoa_parameters()
    measured_circuit = build_qaoa_circuit(best_result.gamma, best_result.beta, measure=True)

    # Ideal shot-based sampling with no device noise.
    ideal_simulator = AerSimulator()
    ideal_circuit = transpile(measured_circuit, ideal_simulator)
    ideal_counts = ideal_simulator.run(ideal_circuit, shots=shots).result().get_counts(0)

    # Device-like noisy simulation using a fake IBM backend noise model.
    backend = FakeManilaV2()
    noisy_simulator = AerSimulator.from_backend(backend)
    optimized_pass_manager = generate_preset_pass_manager(
        backend=backend,
        optimization_level=3,
        layout_method="sabre",
        routing_method="sabre",
        seed_transpiler=123,
    )
    optimized_circuit = optimized_pass_manager.run(measured_circuit)
    noisy_counts = noisy_simulator.run(optimized_circuit, shots=shots).result().get_counts(0)

    elapsed = time.perf_counter() - start_time

    print("=== Package installation ===")
    print("pip install qiskit numpy qiskit-aer qiskit-ibm-runtime")
    print()

    print("=== 4-Qubit Max-Cut simulation setup ===")
    print(f"Qubits: {NUM_QUBITS}")
    print(f"Edges: {EDGES}")
    print(f"Best gamma: {best_result.gamma:.6f}")
    print(f"Best beta:  {best_result.beta:.6f}")
    print(f"Shots: {shots}")
    print()

    print_summary("Ideal Aer simulation", ideal_counts)
    print_summary("Noisy backend-like simulation", noisy_counts)

    print("=== Optimized circuit used for noisy simulation (OpenQASM) ===")
    print(qasm_output(optimized_circuit))
    print()
    print(f"Depth: {optimized_circuit.depth()}")
    print(f"Size: {optimized_circuit.size()}")
    print(f"Operation counts: {dict(optimized_circuit.count_ops())}")
    print()
    print(f"Runtime (seconds): {elapsed:.6f}")


if __name__ == "__main__":
    main()
