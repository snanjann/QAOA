import argparse
import json
import os
import sys

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from qaoa import brute_force_maxcut, rqaoa_solve


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RQAOA for Max-Cut.")
    parser.add_argument("--graph", choices=["complete", "regular"], default="regular")
    parser.add_argument("--nodes", type=int, default=8)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--threshold", type=int, default=2)
    parser.add_argument("--restarts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    if args.graph == "complete":
        graph = nx.complete_graph(args.nodes)
    else:
        graph = nx.random_regular_graph(args.degree, args.nodes, seed=args.seed)

    optimum, _ = brute_force_maxcut(graph)
    assignment, cut, iterations, _ = rqaoa_solve(
        graph,
        p=args.depth,
        n_c=args.threshold,
        n_restarts=args.restarts,
        verbose=False,
    )

    print(json.dumps({
        "experiment": "rqaoa",
        "graph_type": args.graph,
        "seed": args.seed,
        "nodes": args.nodes,
        "degree": args.degree if args.graph == "regular" else None,
        "optimum_cut": optimum,
        "rqaoa_cut": cut,
        "approximation_ratio": cut / optimum,
        "iterations": iterations,
        "assignment": assignment,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
