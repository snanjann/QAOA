"""
qaoa — Qiskit QAOA library for MaxCut.

Two modules:
  core       : graph helpers, Hamiltonian, circuit builders, edge colouring
  algorithms : optimisers, RQAOA, ERQAOA

All public functions re-exported here so notebooks use:
    from qaoa import build_qaoa_circuit, rqaoa_solve, ...
"""
from .core import (
    cut_value, cut_value_bits, uniform_cut_expectation,
    brute_force_maxcut, build_cost_hamiltonian, extract_ising_coefficients,
    build_qaoa_circuit, build_qaoa_circuit_parallel,
    edge_coloring, group_edges_by_colour, circuit_stats,
)
from .algorithms import (
    parameter_shift_gradient, optimise_qaoa, interp_init, AdaptiveShotSchedule,
    compute_correlator, compute_all_correlators,
    eliminate_vertex, solve_small_instance, correlator_sign_accuracy,
    rqaoa_solve,
    select_max_weight_matching, eliminate_vertex_soft, erqaoa_solve,
)
