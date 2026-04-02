"""
qaoa/algorithms.py
==================
All variational algorithms that sit on top of the core circuit/Hamiltonian
primitives.  Merged from:
  - optimizers.py  (parameter optimisation, warm-start, shot scheduling)
  - rqaoa.py       (standard recursive QAOA)

Extended with ERQAOA (multi-edge elimination + soft correlator weighting)
which was previously duplicated inline in erqaoa.ipynb.

Merging these makes sense because ERQAOA shares ~80% of its code with
RQAOA and both depend on the same optimisation helpers — splitting them
created circular-import pressure and forced every notebook to redefine
the shared helpers.

Public API
----------
Optimisation
    parameter_shift_gradient(objective, params)
    optimise_qaoa(objective, n_params, ...)     COBYLA / L-BFGS-B multi-restart
    interp_init(gamma_prev, beta_prev)          INTERP warm-start (Zhou 2020)
    AdaptiveShotSchedule                        shot-doubling for noisy loops

Standard RQAOA
    rqaoa_solve(graph, p, n_c, ...)
    compute_correlator(sv, i, j, n)
    compute_all_correlators(sv, graph, node_to_qubit)
    eliminate_vertex(graph, k, l, s, J, h)
    solve_small_instance(graph, J, h)
    correlator_sign_accuracy(trace, optimal_assignment)

Extended RQAOA (ERQAOA)
    select_max_weight_matching(correlators, k_max)
    eliminate_vertex_soft(graph, k, l, M_kl, J, h)
    erqaoa_solve(graph, p, n_c, k_max, ...)
"""
from __future__ import annotations

from itertools import product
from typing import Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy.optimize import minimize
from qiskit.quantum_info import Statevector, SparsePauliOp

from .core import (
    build_qaoa_circuit,
    build_cost_hamiltonian,
    extract_ising_coefficients,
    brute_force_maxcut,
)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Parameter optimisation
# ═════════════════════════════════════════════════════════════════════════════

def parameter_shift_gradient(
    objective: Callable[[np.ndarray], float],
    params: np.ndarray,
    shift: float = np.pi / 2,
) -> np.ndarray:
    """
    Exact gradient of *objective* at *params* via the parameter-shift rule
    (Mitarai et al. 2018, survey Eq. 3.11):

        dF/dθ_k = ½ [F(…,θ_k+π/2,…) − F(…,θ_k−π/2,…)]

    *objective* returns −F (negative, for minimisation).
    The returned gradient is therefore the gradient of −F.

    Cost: exactly 2 × len(params) circuit evaluations.
    """
    grad = np.zeros_like(params, dtype=float)
    for k in range(len(params)):
        p_plus  = params.copy(); p_plus[k]  += shift
        p_minus = params.copy(); p_minus[k] -= shift
        grad[k] = (objective(p_minus) - objective(p_plus)) / 2.0
    return grad


def optimise_qaoa(
    objective: Callable[[np.ndarray], float],
    n_params: int,
    n_restarts: int   = 5,
    method: str       = "COBYLA",
    maxiter: int      = 500,
    use_gradient: bool = False,
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Minimise *objective* over *n_params* parameters with multiple random
    restarts.  Returns (best_params, best_val, per-restart history).

    Parameters
    ----------
    objective    : f(params) → float  (return −F_p for maximisation)
    n_params     : 2p
    n_restarts   : number of independent random initialisations
    method       : 'COBYLA' (derivative-free, default) or 'L-BFGS-B'
    maxiter      : max iterations per restart
    use_gradient : if True and method='L-BFGS-B', use param-shift gradient
    """
    best_val    = np.inf
    best_params = np.zeros(n_params)
    history: List[float] = []

    for _ in range(n_restarts):
        p0 = np.random.uniform(0.0, 2.0 * np.pi, n_params)

        if method == "COBYLA":
            res = minimize(
                objective, p0, method="COBYLA",
                options={"maxiter": maxiter, "rhobeg": 0.5, "catol": 0.0},
            )
        elif method == "L-BFGS-B":
            jac = (
                (lambda p: parameter_shift_gradient(objective, p))
                if use_gradient else "2-point"
            )
            res = minimize(
                objective, p0, method="L-BFGS-B", jac=jac,
                options={"maxiter": maxiter, "ftol": 1e-9},
            )
        else:
            raise ValueError(f"Unknown method {method!r}. Use 'COBYLA' or 'L-BFGS-B'.")

        history.append(res.fun)
        if res.fun < best_val:
            best_val, best_params = res.fun, res.x.copy()

    return best_params, best_val, history


def interp_init(
    gamma_prev: np.ndarray,
    beta_prev:  np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    INTERP heuristic (Zhou et al. 2020, survey Eq. 3.13).

    Given optimal (γ*, β*) for depth p−1, return a good initialisation
    for depth p by linear interpolation.  Reduces COBYLA iterations by
    3–5× because the interpolated point lies in the basin of the p-optimum.
    """
    p_old = len(gamma_prev)
    p_new = p_old + 1

    def _interp(arr: np.ndarray) -> np.ndarray:
        out = np.zeros(p_new)
        for k in range(1, p_new + 1):
            frac  = (k - 1) * (p_old - 1) / max(p_new - 1, 1)
            lo    = int(np.floor(frac))
            hi    = int(np.ceil(frac))
            lo    = np.clip(lo, 0, p_old - 1)
            hi    = np.clip(hi, 0, p_old - 1)
            alpha = frac - lo
            out[k - 1] = (1 - alpha) * arr[lo] + alpha * arr[hi]
        return out

    return _interp(gamma_prev), _interp(beta_prev)


class AdaptiveShotSchedule:
    """
    Doubles the shot budget when optimisation stalls.

    Use with SPSA or a custom loop (COBYLA has no per-step callback).

        sched = AdaptiveShotSchedule()
        for step in range(max_steps):
            value = noisy_objective(params, shots=sched.shots)
            sched.update(value)
    """

    def __init__(
        self,
        S0: int        = 128,
        patience: int  = 20,
        max_shots: int = 4096,
        tol: float     = 1e-4,
    ) -> None:
        self.shots     = S0
        self._S0       = S0
        self.patience  = patience
        self.max_shots = max_shots
        self._tol      = tol
        self._best     = np.inf
        self._stall    = 0
        self.log: List[int] = []

    def update(self, value: float) -> None:
        self.log.append(self.shots)
        if value < self._best - self._tol:
            self._best  = value
            self._stall = 0
        else:
            self._stall += 1
            if self._stall >= self.patience:
                self.shots  = min(self.shots * 2, self.max_shots)
                self._stall = 0

    def reset(self) -> None:
        self.shots  = self._S0
        self._best  = np.inf
        self._stall = 0
        self.log    = []


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Two-point correlator and variable elimination (shared by RQAOA & ERQAOA)
# ═════════════════════════════════════════════════════════════════════════════

def compute_correlator(
    sv: Statevector,
    i: int,
    j: int,
    n: int,
) -> float:
    """
    M_{ij} = ⟨Z_i Z_j⟩ for a Statevector on n qubits.

    i, j are QUBIT INDICES (0..n-1) — NOT original vertex labels.
    Qiskit convention: qubit 0 = rightmost character in the Pauli string.
    """
    chars    = ["I"] * n
    chars[i] = "Z"
    chars[j] = "Z"
    op = SparsePauliOp("".join(reversed(chars)))
    return float(sv.expectation_value(op).real)


def compute_all_correlators(
    sv: Statevector,
    graph: nx.Graph,
    node_to_qubit: Dict,
) -> Dict[Tuple, float]:
    """
    Compute M_{ij} for every edge in *graph*.

    Returns {(min_v, max_v): M_ij}.

    Uses node_to_qubit to convert vertex label → qubit index, which
    must be rebuilt at every RQAOA iteration after vertex elimination.
    """
    n      = graph.number_of_nodes()
    result = {}
    for u, v in graph.edges():
        qi  = node_to_qubit[u]
        qj  = node_to_qubit[v]
        key = (min(u, v), max(u, v))
        result[key] = compute_correlator(sv, qi, qj, n)
    return result


def _select_max_correlator(
    correlators: Dict[Tuple, float],
) -> Tuple[int, int, int]:
    """Return (k, l, s) for the edge with the largest |M_{ij}|."""
    if not correlators:
        raise ValueError("Correlator dict is empty: graph has no edges.")
    (k, l)  = max(correlators, key=lambda e: abs(correlators[e]))
    m_val   = correlators[(k, l)]
    s       = int(np.sign(m_val)) if abs(m_val) > 1e-10 else 1
    return k, l, s


def eliminate_vertex(
    graph: nx.Graph,
    k: int,
    l: int,
    s: int,
    J: Dict[Tuple, float],
    h: Dict[int, float],
) -> Tuple[nx.Graph, Dict[Tuple, float], Dict[int, float]]:
    """
    Standard (hard) RQAOA elimination: z_k = s · z_l,  s = sgn(M_{kl}).

    Survey Eqs. 5.3–5.4:
        J'_{il} = J_{il} + s · J_{ik}   for i ∉ {k, l}
        h'_l    = h_l   + s · h_k

    The (k,l) edge becomes a constant and is removed from the graph.
    """
    return _eliminate_generic(graph, k, l, float(s), J, h)


def eliminate_vertex_soft(
    graph: nx.Graph,
    k: int,
    l: int,
    M_kl: float,
    J: Dict[Tuple, float],
    h: Dict[int, float],
) -> Tuple[nx.Graph, Dict[Tuple, float], Dict[int, float]]:
    """
    Soft ERQAOA elimination: use raw M_kl (not just sgn) in the update.

        J'_{il} = J_{il} + M_{kl} · J_{ik}     (Eq. 6.4 of paper)
        h'_l    = h_l   + M_{kl} · h_k

    Retains correlator magnitude information through the reduction chain.
    """
    return _eliminate_generic(graph, k, l, M_kl, J, h)


def _eliminate_generic(
    graph: nx.Graph,
    k: int,
    l: int,
    weight: float,
    J: Dict[Tuple, float],
    h: Dict[int, float],
) -> Tuple[nx.Graph, Dict[Tuple, float], Dict[int, float]]:
    """
    Shared logic for hard and soft elimination.  *weight* is either
    sgn(M) = ±1 (RQAOA) or the raw M value (ERQAOA).
    """
    nodes  = [v for v in graph.nodes() if v != k]
    new_h  = {v: h.get(v, 0.0) for v in nodes}
    new_h[l] = h.get(l, 0.0) + weight * h.get(k, 0.0)

    new_J: Dict[Tuple, float] = {}
    for (a, b), w in J.items():
        if k not in (a, b):
            key      = (min(a, b), max(a, b))
            new_J[key] = new_J.get(key, 0.0) + w
        else:
            other = b if a == k else a
            if other == l:
                continue            # (k,l) edge → constant, drop
            key      = (min(l, other), max(l, other))
            new_J[key] = new_J.get(key, 0.0) + weight * w

    # Drop negligible couplings and self-loops
    new_J = {e: w for e, w in new_J.items()
             if abs(w) > 1e-10 and e[0] != e[1]}

    new_graph = nx.Graph()
    new_graph.add_nodes_from(nodes)
    for (a, b), w in new_J.items():
        new_graph.add_edge(a, b, weight=w)

    return new_graph, new_J, new_h


# ─────────────────────────────────────────────────────────────────────────────
# Classical exact solver (for small residual instances)
# ─────────────────────────────────────────────────────────────────────────────

def solve_small_instance(
    graph: nx.Graph,
    J: Dict[Tuple, float],
    h: Dict[int, float],
) -> Dict[int, int]:
    """
    Exact Ising minimisation by exhaustive search.
    Minimises E = −Σ J_{ij} z_i z_j − Σ h_i z_i.
    Feasible for |V| ≤ n_c (default n_c = 4 in RQAOA).
    """
    nodes = list(graph.nodes())
    n     = len(nodes)
    if n == 0:
        return {}
    if n == 1:
        v = nodes[0]
        return {v: 1 if h.get(v, 0.0) >= 0 else -1}

    best_E   = np.inf
    best_z   = {v: 1 for v in nodes}
    for bits in product([-1, 1], repeat=n):
        z = dict(zip(nodes, bits))
        E = sum(-w * z[a] * z[b] for (a, b), w in J.items())
        E += sum(-hv * z.get(v, 1) for v, hv in h.items())
        if E < best_E:
            best_E, best_z = E, dict(z)
    return best_z


# ─────────────────────────────────────────────────────────────────────────────
# Shared RQAOA / ERQAOA inner loop
# ─────────────────────────────────────────────────────────────────────────────

def _qaoa_Fp(params, graph, p, H):
    """Evaluate F_p via exact Statevector. Returns −F_p for minimisation."""
    qc = build_qaoa_circuit(graph, p, bind_params=params)
    sv = Statevector(qc)
    return -sv.expectation_value(H).real


def _unroll_and_cut(
    original_graph: nx.Graph,
    orig_nodes: list,
    inv_map: dict,
    small_z: dict,
    constraints: list,  # [(k, l, s_or_M), ...]  — value used as multiplier
) -> Tuple[Dict[int, int], float]:
    """
    Unroll a constraint stack and compute the achieved cut on the original graph.
    Works for both hard (s=±1) and soft (M∈[-1,+1]) constraints because
    at unroll time we always round to ±1.
    """
    assignment = dict(small_z)
    for k_c, l_c, val in reversed(constraints):
        s_c = int(np.sign(val)) if abs(val) > 1e-10 else 1
        assignment[k_c] = s_c * assignment.get(l_c, 1)

    for v in range(len(orig_nodes)):
        if v not in assignment:
            assignment[v] = 1

    assignment_orig = {inv_map[v]: assignment[v]
                       for v in assignment if v in inv_map}
    for v in orig_nodes:
        if v not in assignment_orig:
            assignment_orig[v] = 1

    cut = sum(
        d.get("weight", 1.0)
        for u, v, d in original_graph.edges(data=True)
        if assignment_orig.get(u, 1) != assignment_orig.get(v, 1)
    )
    return assignment_orig, cut


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Standard RQAOA (Bravyi et al. 2020 / Bae & Lee 2024)
# ═════════════════════════════════════════════════════════════════════════════

def rqaoa_solve(
    graph: nx.Graph,
    p: int           = 1,
    n_c: int         = 4,
    n_restarts: int  = 3,
    optimizer: str   = "COBYLA",
    verbose: bool    = True,
) -> Tuple[Dict[int, int], float, int, List[Dict]]:
    """
    Recursive QAOA for MaxCut (Algorithm 2 of the paper).

    At each step:
      1. Optimise QAOA on the current (smaller) instance.
      2. Evaluate M_{ij} = ⟨Z_i Z_j⟩ for every edge.
      3. Select (k,l) = argmax|M|; impose z_k = sgn(M_{kl}) · z_l.
      4. Reduce J, h (survey Eqs. 5.3–5.4).
      5. Repeat until |V| ≤ n_c, then solve classically.

    Survey Theorem 5.2 (Bae & Lee 2024): achieves ratio 1 on K_{2n}.

    Parameters
    ----------
    graph      : networkx Graph (any vertex labels)
    p          : QAOA depth per call (fixed throughout)
    n_c        : classical threshold
    n_restarts : COBYLA restarts per call
    optimizer  : 'COBYLA' or 'L-BFGS-B'
    verbose    : print per-iteration info

    Returns
    -------
    assignment, cut, n_iters, trace
    """
    if graph.number_of_nodes() == 0:
        raise ValueError("Input graph is empty.")

    original_graph = graph.copy()
    orig_nodes     = list(graph.nodes())
    mapping        = {v: i for i, v in enumerate(orig_nodes)}
    inv_map        = {i: v for v, i in mapping.items()}
    G_curr         = nx.relabel_nodes(graph, mapping)
    J, h           = extract_ising_coefficients(G_curr)
    constraints: List[Tuple] = []
    trace:       List[Dict]  = []
    n_iters = 0

    while G_curr.number_of_nodes() > n_c:
        n_curr = G_curr.number_of_nodes()
        if G_curr.number_of_edges() == 0:
            if verbose:
                print(f"  iter {n_iters+1}: no edges at |V|={n_curr}; stopping.")
            break

        # Rebuild qubit index map for THIS graph state
        nodes_curr    = list(G_curr.nodes())
        node_to_qubit = {v: i for i, v in enumerate(nodes_curr)}

        H_curr = build_cost_hamiltonian(G_curr)

        def obj(params, g=G_curr, pc=p, Hc=H_curr):
            return _qaoa_Fp(params, g, pc, Hc)

        params_opt, neg_Fp, _ = optimise_qaoa(
            obj, n_params=2 * p, n_restarts=n_restarts, method=optimizer,
        )
        Fp = -neg_Fp

        sv          = Statevector(build_qaoa_circuit(G_curr, p, bind_params=params_opt))
        correlators = compute_all_correlators(sv, G_curr, node_to_qubit)
        k, l, s     = _select_max_correlator(correlators)
        M_val       = correlators[(min(k, l), max(k, l))]

        if verbose:
            print(
                f"  iter {n_iters+1}: |V|={n_curr}, |E|={G_curr.number_of_edges()}, "
                f"F_p={Fp:.4f}, eliminate {k}→({k},{l},s={s:+d}), |M|={abs(M_val):.4f}"
            )

        trace.append({
            "iter": n_iters + 1, "n_vertices": n_curr,
            "n_edges": G_curr.number_of_edges(),
            "Fp": Fp, "k": k, "l": l, "s": s, "M_kl": M_val,
            "correlators": dict(correlators),
        })

        constraints.append((k, l, s))
        G_curr, J, h = eliminate_vertex(G_curr, k, l, s, J, h)
        n_iters += 1

    if verbose:
        print(f"  Classical: |V|={G_curr.number_of_nodes()}, "
              f"|E|={G_curr.number_of_edges()}")

    small_z             = solve_small_instance(G_curr, J, h)
    assignment, cut     = _unroll_and_cut(
        original_graph, orig_nodes, inv_map, small_z, constraints
    )
    return assignment, cut, n_iters, trace


def correlator_sign_accuracy(
    trace: List[Dict],
    optimal_assignment: Dict[int, int],
) -> float:
    """
    Fraction of RQAOA steps where sgn(M_{kl}) correctly predicts the
    optimal relative alignment.  Key diagnostic: high → correlators are
    reliable; low → sign errors are propagating.
    """
    if not trace:
        return float("nan")
    correct = 0
    for step in trace:
        k, l, s_pred = step["k"], step["l"], step["s"]
        if k in optimal_assignment and l in optimal_assignment:
            s_true = int(np.sign(optimal_assignment[k] * optimal_assignment[l]))
            if s_true == 0:
                s_true = 1
            if s_pred == s_true:
                correct += 1
    return correct / len(trace)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Extended RQAOA (ERQAOA)
# ═════════════════════════════════════════════════════════════════════════════

def select_max_weight_matching(
    correlators: Dict[Tuple, float],
    k_max: int = 2,
) -> List[Tuple[Tuple, float]]:
    """
    Greedy max-weight matching on high-|M| edges.

    Returns a list of ((u,v), M_uv) pairs forming a matching
    (no two edges share a vertex) with |list| ≤ k_max.

    This is the selection step for ERQAOA multi-edge elimination
    (paper Eq. 6.3).
    """
    sorted_edges = sorted(correlators.items(),
                          key=lambda x: abs(x[1]), reverse=True)
    matching: List[Tuple[Tuple, float]] = []
    used: set = set()

    for (u, v), M in sorted_edges:
        if u not in used and v not in used:
            matching.append(((u, v), M))
            used |= {u, v}
        if len(matching) >= k_max:
            break

    return matching


def erqaoa_solve(
    graph: nx.Graph,
    p: int          = 1,
    n_c: int        = 4,
    k_max: int      = 2,
    n_restarts: int = 3,
    optimizer: str  = "COBYLA",
    verbose: bool   = True,
) -> Tuple[Dict[int, int], float, int, List[Dict]]:
    """
    Extended Recursive QAOA (ERQAOA).

    Two extensions over standard RQAOA:

    1. Multi-edge elimination (Eq. 6.3): eliminate a max-weight matching
       of up to k_max edges per step, roughly halving iteration count for
       dense graphs.

    2. Soft coupling update (Eq. 6.4): use raw M_{kl} (not sgn) in the
       coupling update, retaining magnitude information.  Hard rounding
       only at the final constraint-unrolling step.

    Parameters
    ----------
    k_max   : maximum edges to eliminate per step (default 2)
    others  : same as rqaoa_solve
    """
    if graph.number_of_nodes() == 0:
        raise ValueError("Input graph is empty.")

    original_graph = graph.copy()
    orig_nodes     = list(graph.nodes())
    mapping        = {v: i for i, v in enumerate(orig_nodes)}
    inv_map        = {i: v for v, i in mapping.items()}
    G_curr         = nx.relabel_nodes(graph, mapping)
    J, h           = extract_ising_coefficients(G_curr)
    constraints: List[Tuple] = []
    trace:       List[Dict]  = []
    n_iters = 0

    while G_curr.number_of_nodes() > n_c:
        n_curr = G_curr.number_of_nodes()
        if G_curr.number_of_edges() == 0:
            break

        nodes_curr    = list(G_curr.nodes())
        node_to_qubit = {v: i for i, v in enumerate(nodes_curr)}

        H_curr = build_cost_hamiltonian(G_curr)

        def obj(params, g=G_curr, pc=p, Hc=H_curr):
            return _qaoa_Fp(params, g, pc, Hc)

        params_opt, neg_Fp, _ = optimise_qaoa(
            obj, n_params=2 * p, n_restarts=n_restarts, method=optimizer,
        )
        Fp = -neg_Fp

        sv          = Statevector(build_qaoa_circuit(G_curr, p, bind_params=params_opt))
        correlators = compute_all_correlators(sv, G_curr, node_to_qubit)
        matching    = select_max_weight_matching(correlators, k_max=k_max)

        if verbose:
            edge_str = ", ".join(f"{u}↔{v}(M={M:.3f})"
                                  for (u, v), M in matching)
            print(
                f"  iter {n_iters+1}: |V|={n_curr}, F_p={Fp:.4f}, "
                f"eliminate [{edge_str}]"
            )

        trace.append({
            "iter": n_iters + 1, "n_vertices": n_curr, "Fp": Fp,
            "n_eliminated": len(matching),
            "matching": [((u, v), M) for (u, v), M in matching],
        })

        # Eliminate all edges in the matching (soft update)
        for (k, l), M_kl in matching:
            if G_curr.has_node(k) and G_curr.has_node(l):
                constraints.append((k, l, M_kl))
                G_curr, J, h = eliminate_vertex_soft(G_curr, k, l, M_kl, J, h)

        n_iters += 1

    if verbose:
        print(f"  Classical: |V|={G_curr.number_of_nodes()}")

    small_z             = solve_small_instance(G_curr, J, h)
    assignment, cut     = _unroll_and_cut(
        original_graph, orig_nodes, inv_map, small_z, constraints
    )
    return assignment, cut, n_iters, trace
