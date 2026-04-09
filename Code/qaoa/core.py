"""
qaoa/core.py
============
All stateless building-block functions for QAOA on MaxCut.

Merged from hamiltonians.py (graph → Hamiltonian, brute-force solver, cut
helpers) and circuit.py (QAOA circuit builders, edge colouring).  Keeping
these together avoids a circular-import chain and makes the paper's
listing-by-listing narrative cleaner: every concept used in the pseudocode
is defined once, in one place, and imported by every notebook.

Public API
----------
Graph helpers
    cut_value(assignment, graph)            weighted cut for any bipartition
    cut_value_bits(bits, graph)             cut for a 0/1 bitstring
    brute_force_maxcut(graph)               exact solver, n ≤ 22
    uniform_cut_expectation(graph)          C̄ = Σ_e w_e / 2

Hamiltonian
    build_cost_hamiltonian(graph)           SparsePauliOp  H_C

Ising coefficients
    extract_ising_coefficients(graph)       J, h dicts

Circuit builders  (vertex re-indexed internally; safe for any label set)
    build_qaoa_circuit(graph, p, ...)       sequential cost layer
    build_qaoa_circuit_parallel(graph, p, ...)  edge-coloured parallel layer

Edge colouring
    edge_coloring(graph)                    {(u,v): colour_int}
    group_edges_by_colour(coloring)         {colour: [(u,v,w)]}

Circuit statistics
    circuit_stats(qc)                       depth, 2Q count, op dict
"""
from __future__ import annotations

from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


# ─────────────────────────────────────────────────────────────────────────────
# Cut-value helpers
# ─────────────────────────────────────────────────────────────────────────────

def cut_value(assignment: Dict, graph: nx.Graph) -> float:
    """
    Weighted cut for a bipartition encoded as assignment = {vertex: 0 or 1}.
    Works with {0,1} or {-1,+1} spin encodings (anything where xi ≠ xj).
    """
    return sum(
        data.get("weight", 1.0)
        for i, j, data in graph.edges(data=True)
        if assignment.get(i, 0) != assignment.get(j, 0)
    )


def cut_value_bits(bits: Sequence[int], graph: nx.Graph) -> float:
    """
    Weighted cut for a 0/1 bitstring indexed by vertex position
    (vertex i → bits[i], vertices ordered as list(graph.nodes())).
    """
    nodes = list(graph.nodes())
    idx   = {v: i for i, v in enumerate(nodes)}
    return sum(
        graph[u][v].get("weight", 1.0)
        for u, v in graph.edges()
        if bits[idx[u]] != bits[idx[v]]
    )


def uniform_cut_expectation(graph: nx.Graph) -> float:
    """C̄ = E_{z~Uniform}[C(z)] = (Σ_e w_e) / 2."""
    return sum(d.get("weight", 1.0) for _, _, d in graph.edges(data=True)) / 2


# ─────────────────────────────────────────────────────────────────────────────
# Brute-force exact MaxCut
# ─────────────────────────────────────────────────────────────────────────────

def brute_force_maxcut(
    graph: nx.Graph,
    max_n: int = 22,
) -> Tuple[float, tuple]:
    """
    Exact MaxCut by exhaustive search over all 2^n bipartitions.

    Feasible for n ≤ max_n (default 22; at n=22 there are 4M partitions).

    Returns (opt_value, opt_bits) where opt_bits is a tuple of 0/1
    in the same order as list(graph.nodes()).
    """
    nodes = list(graph.nodes())
    n     = len(nodes)
    if n > max_n:
        raise ValueError(
            f"Graph has {n} vertices; brute force is limited to {max_n}. "
            "Use a heuristic for larger instances."
        )
    idx       = {v: i for i, v in enumerate(nodes)}
    best_val  = 0.0
    best_bits = tuple([0] * n)

    for bits in product([0, 1], repeat=n):
        val = sum(
            graph[u][v].get("weight", 1.0)
            for u, v in graph.edges()
            if bits[idx[u]] != bits[idx[v]]
        )
        if val > best_val:
            best_val, best_bits = val, bits

    return best_val, best_bits


# ─────────────────────────────────────────────────────────────────────────────
# Ising coefficients
# ─────────────────────────────────────────────────────────────────────────────

def extract_ising_coefficients(
    graph: nx.Graph,
) -> Tuple[Dict[Tuple[int, int], float], Dict[int, float]]:
    """
    Extract J (coupling) and h (local field) dicts from a graph.

    For MaxCut: h_i = 0 for all i, J_{ij} = w_{ij}.
    Edge keys are canonical (min(i,j), max(i,j)).
    """
    J: Dict[Tuple[int, int], float] = {}
    for i, j, data in graph.edges(data=True):
        key   = (min(i, j), max(i, j))
        J[key] = J.get(key, 0.0) + data.get("weight", 1.0)
    h: Dict[int, float] = {v: 0.0 for v in graph.nodes()}
    return J, h


# ─────────────────────────────────────────────────────────────────────────────
# Cost Hamiltonian
# ─────────────────────────────────────────────────────────────────────────────

def build_cost_hamiltonian(graph: nx.Graph) -> SparsePauliOp:
    """
    H_C = (1/2) Σ_{(i,j)∈E} w_{ij} (I − Z_i Z_j)

    VERTEX-SAFE: vertices are re-indexed 0..n-1 internally, so graphs
    with non-contiguous labels (e.g. {1,3,5} after RQAOA elimination)
    never cause IndexError or CircuitError.

    Qiskit convention: qubit 0 is the RIGHTMOST character in the Pauli string.
    """
    n     = graph.number_of_nodes()
    if n == 0:
        raise ValueError("Graph has no vertices.")
    nodes = list(graph.nodes())
    idx   = {v: i for i, v in enumerate(nodes)}   # any labels → 0..n-1

    pauli_list: List[Tuple[str, complex]] = []
    for u, v, data in graph.edges(data=True):
        w    = data.get("weight", 1.0)
        i, j = idx[u], idx[v]
        # + w/2 · I
        pauli_list.append(("I" * n, w / 2))
        # − w/2 · Z_i Z_j
        chars    = ["I"] * n
        chars[i] = "Z"
        chars[j] = "Z"
        pauli_list.append(("".join(reversed(chars)), -w / 2))

    return SparsePauliOp.from_list(pauli_list).simplify()


# ─────────────────────────────────────────────────────────────────────────────
# Edge colouring
# ─────────────────────────────────────────────────────────────────────────────

def edge_coloring(graph: nx.Graph) -> Dict[Tuple, int]:
    """
    Proper edge colouring: {(u,v): colour_int}.

    METHOD: build the line graph L(G) (vertices = edges of G, adjacent
    when they share an endpoint) then apply greedy vertex colouring.
    By Vizing's theorem: colours used ≤ max_degree + 1.

    NOTE: nx.edge_coloring does NOT exist in NetworkX.  This is the
    correct implementation.
    """
    if graph.number_of_edges() == 0:
        return {}
    LG  = nx.line_graph(graph)
    col = nx.coloring.greedy_color(LG, strategy="largest_first")
    return {edge: colour for edge, colour in col.items()}


def group_edges_by_colour(
    coloring: Dict[Tuple, int],
    graph: Optional[nx.Graph] = None,
) -> Dict[int, List[Tuple[int, int, float]]]:
    """
    Invert {(u,v): colour} → {colour: [(qi, qj, w), ...]}.

    If graph is provided, weights are read from it.
    Each colour class is a matching (disjoint qubit pairs).
    """
    groups: Dict[int, List] = {}
    for (u, v), colour in coloring.items():
        w = graph[u][v].get("weight", 1.0) if graph is not None else 1.0
        groups.setdefault(colour, []).append((u, v, w))
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper: re-index vertices 0..n-1
# ─────────────────────────────────────────────────────────────────────────────

def _reindex(graph: nx.Graph) -> Tuple[Dict, List[Tuple[int, int, float]]]:
    """
    Return (node_to_qubit, qubit_edges) where qubit_edges = [(qi,qj,w), ...].
    Safe for any vertex label set.
    """
    nodes         = list(graph.nodes())
    node_to_qubit = {v: i for i, v in enumerate(nodes)}
    qubit_edges   = [
        (node_to_qubit[u], node_to_qubit[v], data.get("weight", 1.0))
        for u, v, data in graph.edges(data=True)
    ]
    return node_to_qubit, qubit_edges


# ─────────────────────────────────────────────────────────────────────────────
# QAOA circuit builders
# ─────────────────────────────────────────────────────────────────────────────

def build_qaoa_circuit(
    graph: nx.Graph,
    p: int,
    bind_params: Optional[Sequence[float]] = None,
) -> QuantumCircuit:
    """
    Depth-p QAOA circuit for MaxCut.

    |ψ_p(γ,β)⟩ = ∏_{k=1}^p  e^{-iβ_k H_B}  e^{-iγ_k H_C}  |+⟩^⊗n

    Cost layer  : CX(i,j) – Rz(−2γ_k w_ij, j) – CX(i,j)  per edge
    Mixing layer: Rx(2β_k)  per qubit

    Parameters
    ----------
    graph       : networkx Graph (any vertex labels)
    p           : depth ≥ 1
    bind_params : 2p floats [γ_0,...,γ_{p-1}, β_0,...,β_{p-1}]
                  or None to return a ParameterVector circuit

    Returns QuantumCircuit (parametrised or fully bound)
    """
    if p < 1:
        raise ValueError(f"Depth p must be ≥ 1, got {p}.")
    n = graph.number_of_nodes()
    if n == 0:
        raise ValueError("Graph has no vertices.")

    _, qubit_edges = _reindex(graph)
    gamma = ParameterVector("γ", p)
    beta  = ParameterVector("β", p)

    qc = QuantumCircuit(n, name=f"QAOA_p{p}")
    qc.h(range(n))
    for k in range(p):
        for qi, qj, w in qubit_edges:          # cost layer
            qc.cx(qi, qj)
            qc.rz(-2.0 * gamma[k] * w, qj)
            qc.cx(qi, qj)
        for q in range(n):                     # mixing layer
            qc.rx(2.0 * beta[k], q)

    if bind_params is not None:
        bind_params = list(bind_params)
        if len(bind_params) != 2 * p:
            raise ValueError(
                f"Expected {2*p} parameters (γ×{p}, β×{p}), got {len(bind_params)}."
            )
        qc = qc.assign_parameters(dict(zip(list(gamma) + list(beta), bind_params)))

    return qc


def build_qaoa_circuit_parallel(
    graph: nx.Graph,
    p: int,
    bind_params: Optional[Sequence[float]] = None,
) -> QuantumCircuit:
    """
    QAOA circuit with edge-colour-parallelised cost layer.

    Edges sharing a colour class form a matching (disjoint qubit pairs)
    and can be scheduled in a single hardware cycle.  Reduces cost-layer
    depth from O(m) to O(χ'(G)) ≤ d+1 (Vizing's theorem).

    The two-qubit gate COUNT is identical to build_qaoa_circuit; only
    the scheduling depth changes.  Verified by test_parallel_same_state.
    """
    if p < 1:
        raise ValueError(f"Depth p must be ≥ 1, got {p}.")
    n = graph.number_of_nodes()
    if n == 0:
        raise ValueError("Graph has no vertices.")

    node_to_qubit, _ = _reindex(graph)
    col    = edge_coloring(graph)
    groups: Dict[int, List] = {}
    for (u, v), colour in col.items():
        qi = node_to_qubit[u]
        qj = node_to_qubit[v]
        w  = graph[u][v].get("weight", 1.0)
        groups.setdefault(colour, []).append((qi, qj, w))

    gamma = ParameterVector("γ", p)
    beta  = ParameterVector("β", p)

    qc = QuantumCircuit(n, name=f"QAOA_par_p{p}")
    qc.h(range(n))
    for k in range(p):
        for colour in sorted(groups):          # one barrier per colour class
            for qi, qj, w in groups[colour]:
                qc.cx(qi, qj)
                qc.rz(-2.0 * gamma[k] * w, qj)
                qc.cx(qi, qj)
            qc.barrier()
        for q in range(n):
            qc.rx(2.0 * beta[k], q)

    if bind_params is not None:
        bind_params = list(bind_params)
        if len(bind_params) != 2 * p:
            raise ValueError(
                f"Expected {2*p} parameters, got {len(bind_params)}."
            )
        qc = qc.assign_parameters(dict(zip(list(gamma) + list(beta), bind_params)))

    return qc


# ─────────────────────────────────────────────────────────────────────────────
# Circuit statistics
# ─────────────────────────────────────────────────────────────────────────────

def circuit_stats(qc: QuantumCircuit) -> dict:
    """depth, n_qubits, n_parameters, two_qubit_gates count, ops dict."""
    two_q = {"cx", "ecr", "cz", "swap", "iswap", "rzz"}
    ops   = dict(qc.count_ops())
    return {
        "depth":           qc.depth(),
        "n_qubits":        qc.num_qubits,
        "n_parameters":    qc.num_parameters,
        "two_qubit_gates": sum(v for k, v in ops.items() if k in two_q),
        "ops":             ops,
    }
