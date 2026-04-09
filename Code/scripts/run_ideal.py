import argparse
import json
import os
import sys

import networkx as nx
import numpy as np
from qiskit.quantum_info import Statevector

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from qaoa import build_cost_hamiltonian, build_qaoa_circuit, brute_force_maxcut, optimise_qaoa


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ideal statevector QAOA for Max-Cut.")
    parser.add_argument("--nodes", type=int, default=8, help="Number of graph nodes.")
    parser.add_argument("--degree", type=int, default=3, help="Degree for the random regular graph.")
    parser.add_argument("--depth-max", type=int, default=3, help="Maximum QAOA depth to evaluate.")
    parser.add_argument("--restarts", type=int, default=5, help="Number of optimizer restarts per depth.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    graph = nx.random_regular_graph(args.degree, args.nodes, seed=args.seed)
    hamiltonian = build_cost_hamiltonian(graph)
    optimum, _ = brute_force_maxcut(graph)

    results = []
    for depth in range(1, args.depth_max + 1):
        def objective(params):
            qc = build_qaoa_circuit(graph, depth, bind_params=params)
            return -Statevector(qc).expectation_value(hamiltonian).real

        _, neg_value, _ = optimise_qaoa(objective, n_params=2 * depth, n_restarts=args.restarts)
        value = -neg_value
        results.append({
            "depth": depth,
            "expectation": value,
            "approximation_ratio": value / optimum,
        })

    print(json.dumps({
        "experiment": "ideal_qaoa",
        "seed": args.seed,
        "nodes": args.nodes,
        "degree": args.degree,
        "optimum_cut": optimum,
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
