"""
test_qaoa.py — unit tests for the qaoa library.
Run with: python -m pytest test_qaoa.py -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import networkx as nx
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qaoa import (
    build_cost_hamiltonian, brute_force_maxcut,
    cut_value, cut_value_bits,
    extract_ising_coefficients, uniform_cut_expectation,
)
from qaoa import (
    build_qaoa_circuit, build_qaoa_circuit_parallel,
    edge_coloring, circuit_stats,
)
from qaoa import (
    interp_init, parameter_shift_gradient, AdaptiveShotSchedule,
)
from qaoa import (
    compute_correlator, eliminate_vertex,
    rqaoa_solve, correlator_sign_accuracy,
)
from qaoa import (
    select_max_weight_matching, eliminate_vertex_soft, erqaoa_solve,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def triangle():  return nx.complete_graph(3)

@pytest.fixture
def k4():        return nx.complete_graph(4)

@pytest.fixture
def k6():        return nx.complete_graph(6)

@pytest.fixture
def reg3_6():    return nx.random_regular_graph(3, 6, seed=42)

@pytest.fixture
def cycle8():    return nx.cycle_graph(8)


# ── hamiltonians ──────────────────────────────────────────────────────────────

class TestCutValue:
    def test_no_edges(self):
        G = nx.Graph(); G.add_nodes_from([0,1])
        assert cut_value({0:0,1:0}, G) == 0.0

    def test_single_edge_cut(self):
        G = nx.path_graph(2)
        assert cut_value({0:0,1:1}, G) == 1.0
        assert cut_value({0:0,1:0}, G) == 0.0

    def test_triangle(self, triangle):
        assert cut_value({0:0,1:1,2:1}, triangle) == 2.0

    def test_weighted(self):
        G = nx.Graph(); G.add_edge(0,1,weight=3.5)
        assert cut_value({0:0,1:1}, G) == 3.5


class TestBruteForce:
    def test_k4(self, k4):
        val, _ = brute_force_maxcut(k4);  assert val == 4.0

    def test_triangle(self, triangle):
        val, _ = brute_force_maxcut(triangle);  assert val == 2.0

    def test_cycle8(self, cycle8):
        # C_8 alternating partition cuts all 8 edges
        val, bits = brute_force_maxcut(cycle8)
        assert val == 8.0

    def test_too_large(self):
        with pytest.raises(ValueError, match="brute force"):
            brute_force_maxcut(nx.complete_graph(25))


class TestCostHamiltonian:
    def test_eigenvalues_match_cut(self, triangle):
        from qiskit.quantum_info import Operator
        H = build_cost_hamiltonian(triangle)
        H_mat = Operator(H).data
        n = triangle.number_of_nodes()
        for z in range(2**n):
            bits = tuple((z >> i) & 1 for i in range(n))
            ev = H_mat[z, z].real
            assert abs(ev - cut_value_bits(bits, triangle)) < 1e-9

    def test_hermitian(self, k4):
        from qiskit.quantum_info import Operator
        H = build_cost_hamiltonian(k4)
        M = Operator(H).data
        assert np.allclose(M, M.conj().T)

    def test_max_eigenvalue(self, k4):
        from qiskit.quantum_info import Operator
        H = build_cost_hamiltonian(k4)
        eigs = np.linalg.eigvalsh(Operator(H).data)
        assert abs(eigs.max() - 4.0) < 1e-9

    def test_noncontiguous_labels(self):
        """After RQAOA elimination vertices may be {1,3,5}."""
        G = nx.Graph()
        G.add_nodes_from([1,3,5]); G.add_edge(1,3); G.add_edge(3,5)
        H = build_cost_hamiltonian(G)  # must NOT crash
        assert H.num_qubits == 3


class TestUniformExpectation:
    def test_k4(self, k4):    assert abs(uniform_cut_expectation(k4) - 3.0) < 1e-9
    def test_tri(self, triangle): assert abs(uniform_cut_expectation(triangle) - 1.5) < 1e-9


# ── circuit ───────────────────────────────────────────────────────────────────

class TestEdgeColoring:
    def test_triangle_needs_3_colours(self, triangle):
        col = edge_coloring(triangle)
        assert len(set(col.values())) == 3

    def test_k4_is_valid_matching(self, k4):
        from qaoa import group_edges_by_colour
        col = edge_coloring(k4)
        groups = group_edges_by_colour(col, k4)
        for colour, edges in group_edges_by_colour(col).items():
            verts = [v for e in edges for v in e[:2]]
            assert len(verts) == len(set(verts)), f"Colour {colour} not a matching"

    def test_reg3_is_valid(self, reg3_6):
        from qaoa import group_edges_by_colour
        col = edge_coloring(reg3_6)
        for colour, edges in group_edges_by_colour(col).items():
            verts = [v for e in edges for v in e[:2]]
            assert len(verts) == len(set(verts))


class TestBuildQAOACircuit:
    def test_param_count(self, triangle):
        assert build_qaoa_circuit(triangle, p=2).num_parameters == 4

    def test_depth_grows_with_p(self, triangle):
        d1 = build_qaoa_circuit(triangle, p=1).depth()
        d2 = build_qaoa_circuit(triangle, p=2).depth()
        assert d2 > d1

    def test_bound_no_params(self, triangle):
        qc = build_qaoa_circuit(triangle, p=1, bind_params=np.zeros(2))
        assert qc.num_parameters == 0

    def test_wrong_param_count(self, triangle):
        with pytest.raises(ValueError, match="parameters"):
            build_qaoa_circuit(triangle, p=1, bind_params=np.zeros(5))

    def test_parallel_same_state(self, triangle):
        """Parallel and sequential must produce identical measurement distributions."""
        params = np.array([0.4, 0.7])
        sv_seq = Statevector(build_qaoa_circuit(triangle, 1, params))
        sv_par = Statevector(build_qaoa_circuit_parallel(triangle, 1, params))
        assert np.allclose(np.abs(sv_seq.data)**2, np.abs(sv_par.data)**2, atol=1e-9), \
            "Sequential and parallel circuits disagree"

    def test_noncontiguous_labels_safe(self):
        """Graphs from RQAOA have non-contiguous vertex labels."""
        G = nx.Graph()
        G.add_nodes_from([2,5,8]); G.add_edge(2,5,weight=1.5); G.add_edge(5,8)
        qc = build_qaoa_circuit(G, p=1)
        assert qc.num_qubits == 3


# ── optimizers ────────────────────────────────────────────────────────────────

class TestInterpInit:
    def test_length(self):
        g1, b1 = np.array([0.5]), np.array([0.8])
        g2, b2 = interp_init(g1, b1)
        assert len(g2) == 2 and len(b2) == 2

    def test_no_nan(self):
        g, b = np.random.rand(3), np.random.rand(3)
        g4, b4 = interp_init(g, b)
        assert not np.any(np.isnan(g4))

    def test_range_preserved(self):
        g = np.array([0.3, 0.7])
        g3, _ = interp_init(g, g)
        assert g3.min() >= g.min() - 1e-9
        assert g3.max() <= g.max() + 1e-9


class TestParamShiftGradient:
    def test_sinusoidal(self):
        """gradient of −sin(θ) is −cos(θ); param-shift should recover it exactly."""
        def f(params):          # −sin(θ_0) − sin(θ_1)
            return -np.sum(np.sin(params))
        params = np.array([0.3, 1.1])
        grad   = parameter_shift_gradient(f, params)
        # gradient of f = −cos(params)
        assert np.allclose(grad, np.cos(params), atol=1e-9)


class TestAdaptiveShotSchedule:
    def test_doubles_after_patience(self):
        sched = AdaptiveShotSchedule(S0=128, patience=3, max_shots=4096)
        # First call: value 1.0 < inf → best=1.0, stall=0  (improvement)
        sched.update(1.0)
        # Calls 2,3,4: no improvement → stall reaches patience at call 4
        sched.update(1.0); sched.update(1.0); sched.update(1.0)
        assert sched.shots == 256

    def test_caps_at_max(self):
        sched = AdaptiveShotSchedule(S0=2048, patience=1, max_shots=4096)
        sched.update(1.0)   # improvement, stall=0
        sched.update(1.0)   # stall=1 ≥ patience=1 → double → 4096
        assert sched.shots == 4096
        sched.update(1.0)   # already at max
        assert sched.shots == 4096

    def test_resets_on_improvement(self):
        sched = AdaptiveShotSchedule(S0=128, patience=3)
        sched.update(1.0)   # improvement
        sched.update(1.0); sched.update(1.0)  # stall 1,2
        sched.update(0.5)   # improvement → stall resets to 0
        sched.update(0.5); sched.update(0.5)  # stall 1,2  (< 3)
        assert sched.shots == 128


# ── rqaoa ─────────────────────────────────────────────────────────────────────

class TestComputeCorrelator:
    def test_plus_state_zero(self):
        """In |+⟩^⊗n each qubit is in X-eigenstate; ⟨ZiZj⟩ = 0."""
        qc = QuantumCircuit(3); qc.h([0,1,2])
        sv = Statevector(qc)
        assert abs(compute_correlator(sv, 0, 1, 3)) < 1e-9

    def test_bell_state_plus_one(self):
        """In (|00⟩+|11⟩)/√2: ⟨Z0Z1⟩ = +1."""
        qc = QuantumCircuit(2); qc.h(0); qc.cx(0,1)
        sv = Statevector(qc)
        assert abs(compute_correlator(sv, 0, 1, 2) - 1.0) < 1e-9


class TestEliminateVertex:
    def test_k4_coupling_cancels(self, k4):
        J, h = extract_ising_coefficients(k4)
        G2, J2, _ = eliminate_vertex(k4, 0, 1, -1, J, h)
        # J'_{12} = J_{12} + (-1)*J_{02} = 1-1 = 0 → edge removed
        assert not G2.has_edge(1,2) or abs(G2[1][2].get("weight",0)) < 1e-9

    def test_node_count(self, k6):
        J, h = extract_ising_coefficients(k6)
        G2, _, _ = eliminate_vertex(k6, 0, 1, 1, J, h)
        assert G2.number_of_nodes() == k6.number_of_nodes() - 1

    def test_soft_node_count(self, k6):
        J, h = extract_ising_coefficients(k6)
        G2, _, _ = eliminate_vertex_soft(k6, 0, 1, 0.75, J, h)
        assert G2.number_of_nodes() == k6.number_of_nodes() - 1


class TestRQAOA:
    def test_k4_ratio_1(self, k4):
        _, cut, _, _ = rqaoa_solve(k4, p=1, n_c=2, n_restarts=3, verbose=False)
        opt, _ = brute_force_maxcut(k4)
        assert abs(cut - opt) < 1e-6

    def test_k6_ratio_1(self, k6):
        _, cut, _, _ = rqaoa_solve(k6, p=1, n_c=2, n_restarts=3, verbose=False)
        opt, _ = brute_force_maxcut(k6)
        assert cut / opt >= 0.99

    def test_all_vertices_assigned(self, reg3_6):
        assignment, _, _, _ = rqaoa_solve(reg3_6, p=1, n_c=4, n_restarts=2, verbose=False)
        assert set(assignment.keys()) == set(reg3_6.nodes())

    def test_spins_pm1(self, triangle):
        assignment, _, _, _ = rqaoa_solve(triangle, p=1, n_c=2, n_restarts=2, verbose=False)
        assert all(s in (-1,1) for s in assignment.values())

    def test_trace_fields(self, k4):
        _, _, _, trace = rqaoa_solve(k4, p=1, n_c=2, n_restarts=2, verbose=False)
        for step in trace:
            assert "iter" in step and "Fp" in step
            assert "M_kl" in step and abs(step["s"]) == 1

    def test_sign_accuracy_in_range(self, k4):
        opt_val, opt_bits = brute_force_maxcut(k4)
        nodes = list(k4.nodes())
        opt_assignment = {nodes[i]: 1-2*opt_bits[i] for i in range(len(nodes))}
        _, _, _, trace = rqaoa_solve(k4, p=1, n_c=2, n_restarts=3, verbose=False)
        acc = correlator_sign_accuracy(trace, opt_assignment)
        assert 0.0 <= acc <= 1.0

    def test_noncontiguous_graph(self):
        """RQAOA must work on graphs with non-0-based vertex labels."""
        G = nx.complete_graph(4)
        G = nx.relabel_nodes(G, {0:10,1:20,2:30,3:40})
        assignment, cut, _, _ = rqaoa_solve(G, p=1, n_c=2, n_restarts=2, verbose=False)
        assert set(assignment.keys()) == {10,20,30,40}
        assert cut >= 0


class TestERQAOA:
    def test_matching_has_no_shared_vertices(self):
        matching = select_max_weight_matching({
            (0, 1): 0.9,
            (1, 2): 0.8,
            (2, 3): 0.7,
            (4, 5): 0.6,
        }, k_max=2)
        used = set()
        for (u, v), _ in matching:
            assert u not in used
            assert v not in used
            used.update([u, v])

    def test_k4_ratio_1(self, k4):
        _, cut, _, _ = erqaoa_solve(k4, p=1, n_c=2, k_max=2, n_restarts=3, verbose=False)
        opt, _ = brute_force_maxcut(k4)
        assert abs(cut - opt) < 1e-6

    def test_all_vertices_assigned(self, reg3_6):
        assignment, cut, _, _ = erqaoa_solve(reg3_6, p=1, n_c=2, k_max=2, n_restarts=2, verbose=False)
        assert set(assignment.keys()) == set(reg3_6.nodes())
        assert cut >= 0


class TestAerPath:
    def test_aer_simulator_executes_parallel_circuit(self, triangle):
        qiskit_aer = pytest.importorskip("qiskit_aer")
        simulator = qiskit_aer.AerSimulator()
        qc = build_qaoa_circuit_parallel(triangle, p=1, bind_params=np.array([0.3, 0.2]))
        qc.measure_all()
        counts = simulator.run(qc, shots=128).result().get_counts()
        assert sum(counts.values()) == 128
