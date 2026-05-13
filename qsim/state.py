"""
Quantum state vector representation + tensor-based gate application.

Conventions
-----------
- n qubits => state vector of length 2**n, complex dtype.
- Qubit q has integer index 0..n-1. Qubit 0 is the least-significant bit (LSB)
  in the basis-state integer representation, matching Qiskit and most
  quantum-computing textbooks. So for a 3-qubit state, the basis index 5 = 101
  means q2=1, q1=0, q0=1.
- A 2x2 single-qubit gate G acts as: |psi'> = (I (x) ... (x) G (x) ... (x) I) |psi>
  where G is in the slot of the target qubit.
- A 4x4 two-qubit gate U (control, target) is written in the basis ordering
  |q_a q_b> with q_a the first argument's value as the high bit of the 2-bit
  index. So CNOT(control=q_a, target=q_b) is the textbook matrix
      [[1,0,0,0],
       [0,1,0,0],
       [0,0,0,1],
       [0,0,1,0]].

All gate application uses numpy tensor reshape + moveaxis. This avoids
constructing the full 2**n x 2**n unitary, keeping memory at O(2**n) instead
of O(4**n), which is what makes simulating up to ~25 qubits feasible on a
laptop.
"""
from __future__ import annotations

import numpy as np


def zero_state(n_qubits: int) -> np.ndarray:
    """Return the |0...0> state vector for n qubits."""
    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1")
    vec = np.zeros(2 ** n_qubits, dtype=complex)
    vec[0] = 1.0
    return vec


def basis_state(n_qubits: int, k: int) -> np.ndarray:
    """Return the computational basis state |k> for n qubits."""
    if not 0 <= k < 2 ** n_qubits:
        raise ValueError(f"basis index {k} out of range for {n_qubits} qubits")
    vec = np.zeros(2 ** n_qubits, dtype=complex)
    vec[k] = 1.0
    return vec


def _qubit_to_axis(q: int, n: int) -> int:
    """Convert qubit index (q0 = LSB) to numpy tensor axis index.

    When a (2**n,) vector is reshaped to (2,)*n, axis 0 is the MOST significant
    bit, so qubit q (LSB-counted) corresponds to axis n-1-q.
    """
    return n - 1 - q


def apply_single(vec: np.ndarray, gate: np.ndarray, qubit: int) -> np.ndarray:
    """Apply a 2x2 single-qubit gate to `qubit` in n-qubit state `vec`."""
    n = int(round(np.log2(vec.size)))
    if gate.shape != (2, 2):
        raise ValueError(f"single-qubit gate must be 2x2, got {gate.shape}")
    ax = _qubit_to_axis(qubit, n)
    tensor = vec.reshape([2] * n)
    tensor = np.moveaxis(tensor, ax, 0)
    shape = tensor.shape
    flat = tensor.reshape(2, -1)
    new_flat = gate @ flat
    new_tensor = new_flat.reshape(shape)
    new_tensor = np.moveaxis(new_tensor, 0, ax)
    return new_tensor.reshape(-1)


def apply_two_qubit(vec: np.ndarray, gate: np.ndarray, q_a: int, q_b: int) -> np.ndarray:
    """Apply a 4x4 two-qubit gate to (q_a, q_b) in n-qubit state `vec`.

    The gate matrix is in the |q_a q_b> basis (q_a is the high bit of the
    2-bit basis index, q_b is the low bit). So if you pass the textbook CNOT
    matrix with q_a as control and q_b as target, it does the right thing.
    """
    n = int(round(np.log2(vec.size)))
    if gate.shape != (4, 4):
        raise ValueError(f"two-qubit gate must be 4x4, got {gate.shape}")
    if q_a == q_b:
        raise ValueError("two-qubit gate needs distinct qubits")
    ax_a = _qubit_to_axis(q_a, n)
    ax_b = _qubit_to_axis(q_b, n)
    tensor = vec.reshape([2] * n)
    tensor = np.moveaxis(tensor, [ax_a, ax_b], [0, 1])
    shape = tensor.shape
    flat = tensor.reshape(4, -1)
    new_flat = gate @ flat
    new_tensor = new_flat.reshape(shape)
    new_tensor = np.moveaxis(new_tensor, [0, 1], [ax_a, ax_b])
    return new_tensor.reshape(-1)


def apply_multi(vec: np.ndarray, gate: np.ndarray, qubits: list[int]) -> np.ndarray:
    """Apply an arbitrary k-qubit unitary (2**k x 2**k) to `qubits`.

    The gate matrix is in the |q_0 q_1 ... q_{k-1}> basis (qubits[0] is the
    high bit of the 2**k-dim basis index).
    """
    k = len(qubits)
    n = int(round(np.log2(vec.size)))
    if gate.shape != (2 ** k, 2 ** k):
        raise ValueError(f"{k}-qubit gate must be {2**k}x{2**k}, got {gate.shape}")
    if len(set(qubits)) != k:
        raise ValueError("qubit indices must be distinct")
    axes = [_qubit_to_axis(q, n) for q in qubits]
    tensor = vec.reshape([2] * n)
    tensor = np.moveaxis(tensor, axes, list(range(k)))
    shape = tensor.shape
    flat = tensor.reshape(2 ** k, -1)
    new_flat = gate @ flat
    new_tensor = new_flat.reshape(shape)
    new_tensor = np.moveaxis(new_tensor, list(range(k)), axes)
    return new_tensor.reshape(-1)


def probabilities(vec: np.ndarray) -> np.ndarray:
    """Born-rule probabilities for every computational basis state."""
    return np.abs(vec) ** 2


def marginal_prob(vec: np.ndarray, qubit: int, outcome: int) -> float:
    """P(measuring `qubit` and obtaining `outcome` in {0,1})."""
    n = int(round(np.log2(vec.size)))
    if outcome not in (0, 1):
        raise ValueError("outcome must be 0 or 1")
    probs = probabilities(vec).reshape([2] * n)
    other_axes = tuple(i for i in range(n) if i != _qubit_to_axis(qubit, n))
    p = probs.sum(axis=other_axes)
    return float(p[outcome])


def project_measure(
    vec: np.ndarray, qubit: int, outcome: int
) -> np.ndarray:
    """Project the state onto the eigenspace where `qubit` has `outcome`,
    then renormalize. Caller must have already sampled the outcome."""
    n = int(round(np.log2(vec.size)))
    ax = _qubit_to_axis(qubit, n)
    tensor = vec.reshape([2] * n).copy()
    # Zero the opposite-outcome slice
    indexer: list = [slice(None)] * n
    indexer[ax] = 1 - outcome
    tensor[tuple(indexer)] = 0.0
    flat = tensor.reshape(-1)
    norm = np.linalg.norm(flat)
    if norm < 1e-15:
        raise RuntimeError(
            "tried to project onto a zero-probability outcome (numerical instability?)"
        )
    return flat / norm


def state_to_str(vec: np.ndarray, threshold: float = 1e-9) -> str:
    """Pretty-print a state vector in Dirac notation, dropping near-zero terms."""
    n = int(round(np.log2(vec.size)))
    terms = []
    for i, amp in enumerate(vec):
        if abs(amp) < threshold:
            continue
        ket = f"|{i:0{n}b}>"
        a = amp.real
        b = amp.imag
        if abs(b) < threshold:
            coef = f"{a:+.4f}"
        elif abs(a) < threshold:
            coef = f"{b:+.4f}j"
        else:
            coef = f"({a:+.4f}{b:+.4f}j)"
        terms.append(f"{coef}{ket}")
    return " ".join(terms) if terms else "0"


def fidelity(psi: np.ndarray, phi: np.ndarray) -> float:
    """|<psi|phi>|^2 — overlap between two pure states."""
    return float(abs(np.vdot(psi, phi)) ** 2)
