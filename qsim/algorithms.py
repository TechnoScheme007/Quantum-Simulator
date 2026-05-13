"""
Canonical quantum algorithms, implemented on top of the qsim simulator.

Each algorithm is a function that takes parameters, runs on a Simulator,
and returns either a result (e.g., the cracked phase, the searched item)
or the Simulator object for further inspection.

References used:
- Nielsen & Chuang, *Quantum Computation and Quantum Information*
- Mermin, *Quantum Computer Science*
- Original papers: Deutsch 1985, Grover 1996, Shor 1994, Bennett+ 1993
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

from . import gates as G
from .simulator import Simulator
from .state import apply_multi


# ===========================================================================
# Bell + GHZ states (entanglement)
# ===========================================================================
def bell(kind: str = "phi+", seed: Optional[int] = None) -> Simulator:
    """Prepare one of the four Bell states.

        |phi+> = (|00> + |11>)/sqrt(2)
        |phi-> = (|00> - |11>)/sqrt(2)
        |psi+> = (|01> + |10>)/sqrt(2)
        |psi-> = (|01> - |10>)/sqrt(2)
    """
    sim = Simulator(2, seed=seed)
    if kind in ("psi+", "psi-"):
        sim.x(0)
    sim.h(1).cnot(1, 0)   # entangle
    if kind in ("phi-", "psi-"):
        sim.z(1)
    return sim


def ghz(n: int, seed: Optional[int] = None) -> Simulator:
    """Prepare the n-qubit GHZ state (|00...0> + |11...1>) / sqrt(2)."""
    if n < 2:
        raise ValueError("GHZ needs n >= 2")
    sim = Simulator(n, seed=seed)
    sim.h(0)
    for q in range(1, n):
        sim.cnot(0, q)
    return sim


# ===========================================================================
# Quantum teleportation (Bennett, Brassard, Crépeau, Jozsa, Peres, Wootters 1993)
# ===========================================================================
def teleport(alpha: complex, beta: complex,
             seed: Optional[int] = None) -> tuple[Simulator, tuple[int, int]]:
    """Teleport a single-qubit state alpha|0> + beta|1> from qubit 0 to qubit 2.

    Returns (final_simulator, (m0, m1)) where m0, m1 are Alice's classical
    measurement results. After this returns, qubit 2 of the simulator holds
    the original state alpha|0> + beta|1>, while qubits 0 and 1 are
    classical bits (m0 and m1) post-measurement.
    """
    norm = math.sqrt(abs(alpha) ** 2 + abs(beta) ** 2)
    if norm < 1e-12:
        raise ValueError("(alpha, beta) cannot both be zero")
    alpha = complex(alpha) / norm
    beta = complex(beta) / norm

    sim = Simulator(3, seed=seed)
    # Set q0 to alpha|0> + beta|1>:
    # Initial state is |000>. Indices 0=|000>, 1=|001>: q0 is LSB.
    sim.vec = np.zeros(8, dtype=complex)
    sim.vec[0] = alpha  # |000>: q2=0, q1=0, q0=0
    sim.vec[1] = beta   # |001>: q2=0, q1=0, q0=1

    # Create a Bell pair between qubits 1 (Alice) and 2 (Bob)
    sim.h(1)
    sim.cnot(1, 2)

    # Alice's protocol on her two qubits (q0, q1)
    sim.cnot(0, 1)
    sim.h(0)

    # Alice measures
    m0 = sim.measure(0)
    m1 = sim.measure(1)

    # Bob applies the appropriate Pauli correction to q2
    if m1 == 1:
        sim.x(2)
    if m0 == 1:
        sim.z(2)
    return sim, (m0, m1)


# ===========================================================================
# Deutsch-Jozsa algorithm (1992)
# ===========================================================================
def deutsch_jozsa(oracle: Callable[[int], int], n_input: int,
                  seed: Optional[int] = None) -> str:
    """Decide whether `oracle: {0..2^n - 1} -> {0,1}` is constant or balanced
    with a SINGLE quantum query. Returns "constant" or "balanced".

    Classical worst case: 2^(n-1) + 1 queries. Quantum: 1 query.
    The oracle is built as a phase oracle U_f|x> = (-1)^{f(x)} |x>.
    """
    sim = Simulator(n_input, seed=seed)

    # Hadamard on all input qubits
    for q in range(n_input):
        sim.h(q)

    # Apply phase oracle: |x> -> (-1)^{f(x)} |x>
    diag = np.array(
        [(-1) ** oracle(x) for x in range(2 ** n_input)],
        dtype=complex,
    )
    sim.vec = sim.vec * diag

    # Hadamard again on all qubits
    for q in range(n_input):
        sim.h(q)

    # Measure; if all-zero result, function is constant. Otherwise balanced.
    bits = sim.measure_all()
    return "constant" if all(b == 0 for b in bits) else "balanced"


# ===========================================================================
# Grover's search (1996)
# ===========================================================================
def grover(n: int, marked: list[int] | int,
           n_iterations: Optional[int] = None,
           seed: Optional[int] = None) -> Simulator:
    """Run Grover's amplitude-amplification search on n qubits, marking the
    state(s) listed in `marked`. Returns the simulator post-amplification.

    With M marked items out of N=2^n, optimal iteration count is
    floor(pi/4 * sqrt(N/M)).
    """
    if isinstance(marked, int):
        marked = [marked]
    N = 2 ** n
    M = len(marked)
    if M == 0 or M > N:
        raise ValueError("need 1..N marked items")

    if n_iterations is None:
        # Optimal iteration count: floor(pi/4 * sqrt(N/M)). Using round() can
        # over-rotate past the success peak; floor() is the standard choice.
        n_iterations = max(1, int(math.floor((math.pi / 4) * math.sqrt(N / M))))

    sim = Simulator(n, seed=seed)
    # Initial superposition
    for q in range(n):
        sim.h(q)

    for _ in range(n_iterations):
        # 1) Oracle: phase-flip marked states
        for m in marked:
            sim.vec[m] *= -1
        # 2) Diffusion: 2|s><s| - I, where |s> = uniform superposition
        #    = H^n (2|0><0| - I) H^n
        for q in range(n):
            sim.h(q)
        # Apply 2|0><0| - I: flip all amplitudes except |0...0>
        sim.vec = -sim.vec
        sim.vec[0] = -sim.vec[0]
        for q in range(n):
            sim.h(q)

    return sim


# ===========================================================================
# Quantum Fourier Transform
# ===========================================================================
def qft(sim: Simulator, qubits: Optional[list[int]] = None,
        inverse: bool = False) -> Simulator:
    """In-place QFT (or inverse) on the given qubits.

    Convention: this implementation produces the bit-reversed QFT and then
    SWAPs to restore order, matching the Nielsen-Chuang standard.
    """
    if qubits is None:
        qubits = list(range(sim.n_qubits))
    k = len(qubits)

    if inverse:
        # Dagger of (gates + SWAP) is (SWAP + dagger of gates), so SWAP first.
        for i in range(k // 2):
            sim.swap(qubits[i], qubits[k - 1 - i])
        for i in range(k):
            for j in range(i):
                theta = -math.pi / 2 ** (i - j)
                sim.cphase(theta, qubits[j], qubits[i])
            sim.h(qubits[i])
    else:
        # MSB-first iteration (Nielsen-Chuang / Qiskit convention).
        for i in reversed(range(k)):
            sim.h(qubits[i])
            for j in reversed(range(i)):
                theta = math.pi / 2 ** (i - j)
                sim.cphase(theta, qubits[j], qubits[i])
        # SWAP to reverse bit order
        for i in range(k // 2):
            sim.swap(qubits[i], qubits[k - 1 - i])

    return sim


# ===========================================================================
# Quantum Phase Estimation
# ===========================================================================
def phase_estimation(unitary_eigenvalue_phase: float,
                     n_count: int,
                     seed: Optional[int] = None) -> float:
    """Estimate `phase` in [0, 1) given a single-qubit eigenvector of a
    unitary U whose eigenvalue is exp(2*pi*i*phase). Uses `n_count` counting
    qubits. Resolution: 2^-n_count.

    For demonstration we use U = phase gate, eigenvalue on |1> = e^{i*2*pi*phi}.
    The state qubit is prepared in |1>.
    """
    n_total = n_count + 1
    state_qubit = n_count  # last qubit holds the eigenvector
    sim = Simulator(n_total, seed=seed)

    # Prepare |1> on the state qubit (eigenvector of phase gate)
    sim.x(state_qubit)

    # Hadamard the counting register
    for q in range(n_count):
        sim.h(q)

    # Apply controlled-U^{2^k} from counting qubit k onto state qubit.
    # For U = phase(2*pi*phi), U^{2^k} = phase(2*pi*phi*2^k).
    for k in range(n_count):
        angle = 2 * math.pi * unitary_eigenvalue_phase * (2 ** k)
        sim.cphase(angle, k, state_qubit)

    # Inverse QFT on the counting register
    qft(sim, list(range(n_count)), inverse=True)

    # Read the counting register as a binary fraction
    probs = sim.probabilities()
    # Marginalize out the state qubit
    n = sim.n_qubits
    grid = probs.reshape([2] * n)
    # Sum over state_qubit axis
    ax = n - 1 - state_qubit
    marg = grid.sum(axis=ax).reshape(-1)
    measured = int(np.argmax(marg))
    # Bits in counting register form the integer "measured" in 0..2^n_count-1.
    return measured / (2 ** n_count)


# ===========================================================================
# Shor's algorithm (factoring N=15 with a=7 or a=2; demonstration scale)
# ===========================================================================
def _modular_exp_unitary(a: int, N: int, n_work: int) -> np.ndarray:
    """Build the unitary U|y> = |a*y mod N> on `n_work` qubits (small N only).

    For y >= N we leave |y> fixed so the matrix is unitary. This is the same
    convention textbooks use for didactic Shor at small scales.
    """
    dim = 2 ** n_work
    U = np.zeros((dim, dim), dtype=complex)
    for y in range(dim):
        if y < N:
            U[(a * y) % N, y] = 1
        else:
            U[y, y] = 1
    return U


def _continued_fraction(num: int, den: int, max_denom: int) -> tuple[int, int]:
    """Find p/q approximating num/den with q <= max_denom (best rational)."""
    from fractions import Fraction
    f = Fraction(num, den).limit_denominator(max_denom)
    return f.numerator, f.denominator


def shor_period(a: int, N: int, n_count: int = 8,
                seed: Optional[int] = None) -> int:
    """Run the quantum subroutine of Shor's algorithm: estimate the period r
    of f(x) = a^x mod N using QPE on the modular-multiplication unitary.

    Returns the estimated period r.
    """
    n_work = math.ceil(math.log2(N))
    n_total = n_count + n_work
    sim = Simulator(n_total, seed=seed)

    # Initialize work register to |1>
    sim.x(n_count)  # qubit n_count is the LSB of the work register

    # Hadamard the counting register
    for q in range(n_count):
        sim.h(q)

    # Apply controlled-U^{2^k} for U|y> = |a y mod N>
    U_a = _modular_exp_unitary(a, N, n_work)
    work_qubits = list(range(n_count, n_count + n_work))
    for k in range(n_count):
        # U^{2^k}: matrix power
        U_pow = np.linalg.matrix_power(U_a, 2 ** k)
        # Build controlled-U_pow: block-diag(I, U_pow) on (control + work_qubits)
        dim_w = 2 ** n_work
        cU = np.eye(2 * dim_w, dtype=complex)
        cU[dim_w:, dim_w:] = U_pow
        # Apply on [control=k, *work_qubits]
        sim.apply(cU, k, *work_qubits)

    # Inverse QFT on counting register
    qft(sim, list(range(n_count)), inverse=True)

    # Measure counting register
    probs = sim.probabilities()
    # Marginalize work register
    n = sim.n_qubits
    grid = probs.reshape([2] * n)
    # Sum over the n_work LSB axes => axes (n-1-q) for q in work register
    # those axes are 0..n_work-1 in numpy (high bits of int = first axes).
    # actually the work qubits are q in [n_count..n_total-1], whose axes
    # are n-1-q = (n_total-1-q). For q=n_count it's n_work-1; for q=n_total-1
    # it's 0. So sum over axes 0..n_work-1.
    marg = grid.sum(axis=tuple(range(n_work))).reshape(-1)
    marg = marg / marg.sum()
    rng = np.random.default_rng(seed)
    measured = int(rng.choice(2 ** n_count, p=marg))

    if measured == 0:
        return 0

    # Continued-fraction recovery of period r from measured/(2^n_count) ~ s/r
    _, r = _continued_fraction(measured, 2 ** n_count, max_denom=N)
    return r


def shor_factor(N: int, attempts: int = 8,
                seed: Optional[int] = None) -> Optional[tuple[int, int]]:
    """Try to factor a small composite N using Shor's algorithm.

    Returns (p, q) with p*q == N, or None if it failed after `attempts` tries.
    Works reliably for N=15, 21, 35 with the default settings. Larger N
    requires more counting qubits and is exponentially slower to simulate.
    """
    if N % 2 == 0:
        return (2, N // 2)

    rng = np.random.default_rng(seed)
    for _ in range(attempts):
        a = int(rng.integers(2, N))
        if math.gcd(a, N) != 1:
            return (math.gcd(a, N), N // math.gcd(a, N))
        # Pick enough counting qubits: 2 * ceil(log2 N) is the textbook minimum
        n_count = 2 * math.ceil(math.log2(N))
        r = shor_period(a, N, n_count=n_count,
                        seed=int(rng.integers(0, 2 ** 31)))
        if r == 0 or r % 2 != 0:
            continue
        x = pow(a, r // 2, N)
        if x == N - 1:
            continue
        p = math.gcd(x + 1, N)
        q = math.gcd(x - 1, N)
        for f in (p, q):
            if 1 < f < N and N % f == 0:
                return (f, N // f)
    return None
