# 4-Qubit Max-Cut QAOA: Mathematical Pseudocode and Implementation Plan

This folder contains code for the **Max-Cut problem only** on a **4-qubit graph** using a depth-1 QAOA ansatz (`p = 1`).

## Problem Definition

Let the graph be

- `G = (V, E)`
- `V = {0, 1, 2, 3}`
- `E = {(0,1), (1,2), (2,3), (3,0)}`

This is the 4-node cycle graph `C_4`.

For a bitstring `z = z_0 z_1 z_2 z_3` with `z_i in {0,1}`, the Max-Cut objective is

`C(z) = sum_{(i,j) in E} [z_i (1 - z_j) + z_j (1 - z_i)]`

Equivalently,

`C(z) = sum_{(i,j) in E} 1/2 (1 - s_i s_j)`, where `s_i in {+1, -1}`.

The quantum cost Hamiltonian for Max-Cut is

`H_C = sum_{(i,j) in E} 1/2 (I - Z_i Z_j)`

The mixer Hamiltonian is

`H_B = sum_{i in V} X_i`

For depth `p = 1`, the QAOA trial state is

`|gamma, beta> = exp(-i beta H_B) exp(-i gamma H_C) |+>^⊗4`

The optimization target is

`F(gamma, beta) = <gamma, beta| H_C |gamma, beta>`

The classical optimum for this graph is

`max_z C(z) = 4`

with optimal cuts given by the alternating bitstrings `0101` and `1010`.

---

## File 1: Ideal QAOA Circuit

**Output file:** `maxcut_qaoa_ideal.py`

### Goal

Construct the ideal logical QAOA circuit, search over `(gamma, beta)`, and evaluate

`F(gamma, beta) = <H_C>`

exactly from the statevector.

### Mathematical pseudocode

1. Define the 4-qubit Max-Cut graph:
   - `V = {0,1,2,3}`
   - `E = {(0,1), (1,2), (2,3), (3,0)}`
2. Build
   - `H_C = sum_{(i,j) in E} 1/2 (I - Z_i Z_j)`
3. For each `gamma` in `[0, pi]` and each `beta` in `[0, pi/2]`:
   - Prepare `|+>^⊗4` by applying `H` to each qubit.
   - Apply the cost unitary
     - `U_C(gamma) = exp(-i gamma H_C)`
   - For each edge `(i,j)`, implement the Max-Cut phase separator by
     - `CX(i,j)`
     - `RZ(-gamma)` on qubit `j`
     - `CX(i,j)`
   - Apply the mixer unitary
     - `U_B(beta) = exp(-i beta H_B)`
   - For each qubit `k`, implement the mixer with
     - `RX(2 beta)` on qubit `k`
   - Compute the exact expectation value
     - `F(gamma, beta) = <psi(gamma,beta)| H_C |psi(gamma,beta)>`
4. Choose
   - `(gamma*, beta*) = argmax F(gamma, beta)`
5. Build the final measured circuit using `(gamma*, beta*)`.
6. Report:
   - `gamma*`
   - `beta*`
   - `F(gamma*, beta*)`
   - approximation ratio `F(gamma*, beta*) / 4`
   - most likely basis state
   - cut value of that basis state
   - total runtime

---

## File 2: Transpilation Comparison

**Output file:** `maxcut_qaoa_transpile.py`

### Goal

Map the ideal Max-Cut circuit to a realistic hardware model and compare a naive transpilation with an optimized one.

### Mathematical / algorithmic pseudocode

1. Reuse the same 4-qubit Max-Cut graph and best QAOA parameters `(gamma*, beta*)`.
2. Build the measured logical circuit
   - `U(gamma*, beta*) = U_B(beta*) U_C(gamma*) H^⊗4`
3. Choose a fake backend, for example `FakeManilaV2`.
4. Baseline transpilation:
   - `T_base = transpile(U, backend, optimization_level = 0)`
5. Optimized transpilation:
   - generate preset pass manager with
     - `optimization_level = 3`
     - `layout_method = "sabre"`
     - `routing_method = "sabre"`
   - `T_opt = PM(U)`
6. Measure for each circuit:
   - depth
   - total size
   - number of two-qubit gates
   - gate histogram
7. Compare
   - `depth(T_base)` vs `depth(T_opt)`
   - `two_qubit(T_base)` vs `two_qubit(T_opt)`
   - `size(T_base)` vs `size(T_opt)`
8. Print both transpiled circuits in OpenQASM form.
9. Report total runtime.

---

## File 3: Simulation

**Output file:** `maxcut_qaoa_simulation.py`

### Goal

Simulate the 4-qubit Max-Cut circuit in both ideal and noisy settings and convert measured bitstrings into cut values.

### Mathematical / algorithmic pseudocode

1. Reuse the graph `G`, the Hamiltonian `H_C`, and the best parameters `(gamma*, beta*)`.
2. Build the measured circuit `U(gamma*, beta*)`.
3. Ideal simulation:
   - use `AerSimulator()`
   - run the circuit for `N` shots
   - obtain counts `p_ideal(z)`
4. Noisy simulation:
   - choose `FakeManilaV2`
   - construct `AerSimulator.from_backend(backend)`
   - transpile the circuit with the optimized pass manager
   - run for `N` shots
   - obtain counts `p_noisy(z)`
5. For every sampled bitstring `z`, compute its Max-Cut value
   - `C(z) = sum_{(i,j) in E} [z_i != z_j]`
6. From the counts, compute:
   - most frequent bitstring
   - cut value of the most frequent bitstring
   - best sampled cut
   - all bitstrings achieving the best sampled cut
   - average sampled cut
7. Compare ideal and noisy sampled performance.
8. Report total runtime.

---

## Practical Implementation Notes

- The implementation is restricted to **Max-Cut only**.
- The graph is fixed to **4 qubits / 4 vertices**.
- The ideal script performs an exact statevector evaluation.
- The transpilation script studies hardware-aware compilation.
- The simulation script studies sampling under ideal and noisy execution.
- The Python scripts print **OpenQASM** instead of Unicode circuit drawings so they run safely in Windows terminals.

---

## Required Packages

Install these before running the Python files:

```bash
pip install qiskit numpy qiskit-aer qiskit-ibm-runtime
```

If `pip` maps to another Python installation on your system, use:

```bash
python -m pip install qiskit numpy qiskit-aer qiskit-ibm-runtime
```
