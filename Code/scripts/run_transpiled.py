import argparse
import json
import os
import sys

import networkx as nx
import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from qaoa import (
    build_cost_hamiltonian,
    build_qaoa_circuit_parallel,
    brute_force_maxcut,
    interp_init,
    optimise_qaoa,
    uniform_cut_expectation,
)


def build_noise_model(eps_1q: float, eps_2q: float) -> NoiseModel:
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(eps_1q, 1), ["rz", "rx", "h", "x", "sx"])
    noise_model.add_all_qubit_quantum_error(depolarizing_error(eps_2q, 2), ["cx", "ecr"])
    return noise_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run transpiled noisy Aer QAOA for Max-Cut.")
    parser.add_argument("--nodes", type=int, default=6)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--depth-max", type=int, default=3)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--eps-1q", type=float, default=1e-4)
    parser.add_argument("--eps-2q", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    graph = nx.random_regular_graph(args.degree, args.nodes, seed=args.seed)
    hamiltonian = build_cost_hamiltonian(graph)
    optimum, _ = brute_force_maxcut(graph)
    baseline = uniform_cut_expectation(graph)
    simulator = AerSimulator(noise_model=build_noise_model(args.eps_1q, args.eps_2q))
    nodes = list(graph.nodes())

    gamma_prev = None
    beta_prev = None
    results = []

    for depth in range(1, args.depth_max + 1):
        def objective(params):
            qc = build_qaoa_circuit_parallel(graph, depth, bind_params=params)
            return -Statevector(qc).expectation_value(hamiltonian).real

        if gamma_prev is not None:
            g0, b0 = interp_init(gamma_prev, beta_prev)
            init = np.concatenate([g0, b0])

            def objective_with_init(params):
                return objective(params)

            params_opt, neg_value, _ = optimise_qaoa(
                objective_with_init,
                n_params=2 * depth,
                n_restarts=max(args.restarts - 1, 1),
            )
        else:
            params_opt, neg_value, _ = optimise_qaoa(objective, n_params=2 * depth, n_restarts=args.restarts)

        ideal_value = -neg_value
        gamma_prev = params_opt[:depth]
        beta_prev = params_opt[depth:]

        qc = build_qaoa_circuit_parallel(graph, depth, bind_params=params_opt)
        qc_t = transpile(qc, basis_gates=["cx", "rz", "rx", "h", "x"], optimization_level=3)
        gate_count = qc_t.count_ops().get("cx", 0) + qc_t.count_ops().get("ecr", 0)

        measured = qc_t.copy()
        measured.measure_all()
        counts = simulator.run(measured, shots=args.shots).result().get_counts()
        total = sum(counts.values())
        noisy_value = 0.0
        for bitstring, count in counts.items():
            assignment = {nodes[i]: int(bit) for i, bit in enumerate(reversed(bitstring[: graph.number_of_nodes()]))}
            cut = sum(graph[u][v].get("weight", 1.0) for u, v in graph.edges() if assignment[u] != assignment[v])
            noisy_value += count / total * cut

        predicted = (1 - args.eps_2q) ** gate_count * ideal_value + (1 - (1 - args.eps_2q) ** gate_count) * baseline
        results.append({
            "depth": depth,
            "ideal_expectation": ideal_value,
            "noisy_expectation": noisy_value,
            "predicted_expectation": predicted,
            "two_qubit_gate_count": gate_count,
            "ideal_ratio": ideal_value / optimum,
            "noisy_ratio": noisy_value / optimum,
        })

    print(json.dumps({
        "experiment": "transpiled_aer_qaoa",
        "seed": args.seed,
        "nodes": args.nodes,
        "degree": args.degree,
        "optimum_cut": optimum,
        "uniform_baseline": baseline,
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
