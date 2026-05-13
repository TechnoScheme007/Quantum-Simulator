"""
Standard quantum gate library.

All matrices are numpy complex arrays. Single-qubit gates are 2x2. Two-qubit
gates are 4x4 in the |q_a q_b> basis (q_a is the high bit of the 2-bit
basis index). Three-qubit gates are 8x8 in the |q_a q_b q_c> basis.

Parameterized gate functions take the angle in radians.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Single-qubit gates
# ---------------------------------------------------------------------------
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = (1.0 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
S = np.array([[1, 0], [0, 1j]], dtype=complex)          # phase pi/2
Sdg = np.array([[1, 0], [0, -1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)  # pi/4
Tdg = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex)


def Rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def Ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def Rz(theta: float) -> np.ndarray:
    return np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]],
        dtype=complex,
    )


def phase(theta: float) -> np.ndarray:
    """P(theta) = diag(1, e^{i theta}). Global-phase-free version of Rz."""
    return np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex)


def U3(theta: float, phi: float, lam: float) -> np.ndarray:
    """General single-qubit unitary (IBM Qiskit's U3 convention)."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array(
        [
            [c, -np.exp(1j * lam) * s],
            [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c],
        ],
        dtype=complex,
    )


# ---------------------------------------------------------------------------
# Two-qubit gates
# ---------------------------------------------------------------------------
# CNOT with q_a as control, q_b as target. Basis |q_a q_b>.
CNOT = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ],
    dtype=complex,
)

# CZ: phase-flip when both qubits are |1>
CZ = np.diag([1, 1, 1, -1]).astype(complex)

# SWAP
SWAP = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=complex,
)

# iSWAP
ISWAP = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, 1j, 0],
        [0, 1j, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=complex,
)


def controlled(U: np.ndarray) -> np.ndarray:
    """Promote a k-qubit unitary U to a controlled-U on k+1 qubits.

    Control is the first (high-order) qubit. Result is 2*dim x 2*dim:
        [[I, 0],
         [0, U]]
    """
    dim = U.shape[0]
    out = np.eye(2 * dim, dtype=complex)
    out[dim:, dim:] = U
    return out


def CRz(theta: float) -> np.ndarray:
    return controlled(Rz(theta))


def CPhase(theta: float) -> np.ndarray:
    return controlled(phase(theta))


# ---------------------------------------------------------------------------
# Three-qubit gates
# ---------------------------------------------------------------------------
# Toffoli (CCX) with q_a, q_b as controls, q_c as target.
TOFFOLI = np.eye(8, dtype=complex)
TOFFOLI[[6, 7]] = TOFFOLI[[7, 6]]  # swap the |110> and |111> rows

# Fredkin (CSWAP): q_a controls SWAP of (q_b, q_c)
FREDKIN = np.eye(8, dtype=complex)
FREDKIN[[5, 6]] = FREDKIN[[6, 5]]  # swap |101> and |110>


# ---------------------------------------------------------------------------
# Sanity checks (will run on import; cheap)
# ---------------------------------------------------------------------------
def _is_unitary(U: np.ndarray, tol: float = 1e-10) -> bool:
    n = U.shape[0]
    return np.allclose(U @ U.conj().T, np.eye(n), atol=tol)


for _name, _U in [
    ("X", X), ("Y", Y), ("Z", Z), ("H", H), ("S", S), ("T", T),
    ("CNOT", CNOT), ("CZ", CZ), ("SWAP", SWAP), ("ISWAP", ISWAP),
    ("TOFFOLI", TOFFOLI), ("FREDKIN", FREDKIN),
]:
    assert _is_unitary(_U), f"{_name} is not unitary"
