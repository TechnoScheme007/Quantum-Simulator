"""
Quantum simulator: state-vector backend with a Circuit DSL.

Two ways to drive it:

    # Direct, imperative
    sim = Simulator(2)
    sim.h(0)
    sim.cnot(0, 1)
    print(sim.sample(shots=1024))

    # Declarative — build, then run
    qc = Circuit(2)
    qc.h(0).cnot(0, 1).measure_all()
    print(qc.run(shots=1024))
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import gates as G
from . import state as S


@dataclass
class Simulator:
    n_qubits: int
    seed: Optional[int] = None
    vec: np.ndarray = field(init=False)
    rng: np.random.Generator = field(init=False)
    measurements: list = field(default_factory=list, init=False)

    def __post_init__(self):
        self.vec = S.zero_state(self.n_qubits)
        self.rng = np.random.default_rng(self.seed)

    # ---------- low-level apply ----------
    def apply(self, U: np.ndarray, *qubits: int) -> "Simulator":
        """Apply an arbitrary k-qubit unitary to the given qubits."""
        if len(qubits) == 1:
            self.vec = S.apply_single(self.vec, U, qubits[0])
        elif len(qubits) == 2:
            self.vec = S.apply_two_qubit(self.vec, U, qubits[0], qubits[1])
        else:
            self.vec = S.apply_multi(self.vec, U, list(qubits))
        return self

    # ---------- single-qubit gates ----------
    def i(self, q: int) -> "Simulator": return self.apply(G.I, q)
    def x(self, q: int) -> "Simulator": return self.apply(G.X, q)
    def y(self, q: int) -> "Simulator": return self.apply(G.Y, q)
    def z(self, q: int) -> "Simulator": return self.apply(G.Z, q)
    def h(self, q: int) -> "Simulator": return self.apply(G.H, q)
    def s(self, q: int) -> "Simulator": return self.apply(G.S, q)
    def sdg(self, q: int) -> "Simulator": return self.apply(G.Sdg, q)
    def t(self, q: int) -> "Simulator": return self.apply(G.T, q)
    def tdg(self, q: int) -> "Simulator": return self.apply(G.Tdg, q)
    def rx(self, theta: float, q: int) -> "Simulator": return self.apply(G.Rx(theta), q)
    def ry(self, theta: float, q: int) -> "Simulator": return self.apply(G.Ry(theta), q)
    def rz(self, theta: float, q: int) -> "Simulator": return self.apply(G.Rz(theta), q)
    def p(self, theta: float, q: int) -> "Simulator": return self.apply(G.phase(theta), q)
    def u3(self, theta: float, phi: float, lam: float, q: int) -> "Simulator":
        return self.apply(G.U3(theta, phi, lam), q)

    # ---------- two-qubit gates ----------
    def cnot(self, control: int, target: int) -> "Simulator":
        return self.apply(G.CNOT, control, target)

    def cx(self, control: int, target: int) -> "Simulator":
        return self.cnot(control, target)

    def cz(self, q_a: int, q_b: int) -> "Simulator":
        return self.apply(G.CZ, q_a, q_b)

    def swap(self, q_a: int, q_b: int) -> "Simulator":
        return self.apply(G.SWAP, q_a, q_b)

    def iswap(self, q_a: int, q_b: int) -> "Simulator":
        return self.apply(G.ISWAP, q_a, q_b)

    def crz(self, theta: float, control: int, target: int) -> "Simulator":
        return self.apply(G.CRz(theta), control, target)

    def cphase(self, theta: float, control: int, target: int) -> "Simulator":
        return self.apply(G.CPhase(theta), control, target)

    # ---------- three-qubit gates ----------
    def toffoli(self, c1: int, c2: int, target: int) -> "Simulator":
        return self.apply(G.TOFFOLI, c1, c2, target)

    def ccx(self, c1: int, c2: int, target: int) -> "Simulator":
        return self.toffoli(c1, c2, target)

    def fredkin(self, control: int, q_a: int, q_b: int) -> "Simulator":
        return self.apply(G.FREDKIN, control, q_a, q_b)

    def cswap(self, control: int, q_a: int, q_b: int) -> "Simulator":
        return self.fredkin(control, q_a, q_b)

    # ---------- measurement ----------
    def measure(self, q: int) -> int:
        """Projective measurement of one qubit. Collapses the state. Returns 0 or 1."""
        p0 = S.marginal_prob(self.vec, q, 0)
        outcome = 0 if self.rng.random() < p0 else 1
        self.vec = S.project_measure(self.vec, q, outcome)
        self.measurements.append((q, outcome))
        return outcome

    def measure_all(self) -> list[int]:
        """Measure every qubit in order; collapses the state."""
        return [self.measure(q) for q in range(self.n_qubits)]

    # ---------- sampling without collapse ----------
    def probabilities(self) -> np.ndarray:
        """Full Born-rule probability distribution over computational basis."""
        return S.probabilities(self.vec)

    def sample(self, shots: int = 1024) -> dict[str, int]:
        """Sample shots without collapsing the state.

        Returns a dict mapping bitstrings (q_{n-1}...q_0) to counts.
        """
        probs = self.probabilities()
        # numpy may accumulate tiny imag/negatives — clean
        probs = np.maximum(probs.real, 0.0)
        probs = probs / probs.sum()
        outcomes = self.rng.choice(probs.size, size=shots, p=probs)
        counts: dict[str, int] = {}
        for o in outcomes:
            key = f"{o:0{self.n_qubits}b}"
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def expectation(self, observable: np.ndarray) -> float:
        """<psi | observable | psi> for a Hermitian observable matrix."""
        expected_dim = 2 ** self.n_qubits
        if observable.shape != (expected_dim, expected_dim):
            raise ValueError(
                f"observable must be {expected_dim}x{expected_dim}, got {observable.shape}"
            )
        return float(np.real(np.vdot(self.vec, observable @ self.vec)))

    # ---------- state inspection ----------
    def state_str(self, threshold: float = 1e-9) -> str:
        return S.state_to_str(self.vec, threshold=threshold)

    def reset(self) -> "Simulator":
        """Reset to |0...0> without changing seed."""
        self.vec = S.zero_state(self.n_qubits)
        self.measurements = []
        return self

    def copy(self) -> "Simulator":
        out = Simulator(self.n_qubits)
        out.vec = self.vec.copy()
        out.rng = np.random.default_rng()  # fresh
        out.measurements = list(self.measurements)
        return out


# ---------------------------------------------------------------------------
# Declarative circuit builder
# ---------------------------------------------------------------------------
@dataclass
class Circuit:
    """A circuit you build first, run later. Same API as Simulator but the
    operations are recorded and replayed on `.run(...)`."""
    n_qubits: int
    ops: list = field(default_factory=list)

    def _add(self, name, *args):
        self.ops.append((name, args))
        return self

    # forward all gate methods as recorders
    def i(self, q): return self._add("i", q)
    def x(self, q): return self._add("x", q)
    def y(self, q): return self._add("y", q)
    def z(self, q): return self._add("z", q)
    def h(self, q): return self._add("h", q)
    def s(self, q): return self._add("s", q)
    def sdg(self, q): return self._add("sdg", q)
    def t(self, q): return self._add("t", q)
    def tdg(self, q): return self._add("tdg", q)
    def rx(self, theta, q): return self._add("rx", theta, q)
    def ry(self, theta, q): return self._add("ry", theta, q)
    def rz(self, theta, q): return self._add("rz", theta, q)
    def p(self, theta, q): return self._add("p", theta, q)
    def u3(self, t, p, l, q): return self._add("u3", t, p, l, q)
    def cnot(self, c, t): return self._add("cnot", c, t)
    def cx(self, c, t): return self._add("cnot", c, t)
    def cz(self, a, b): return self._add("cz", a, b)
    def swap(self, a, b): return self._add("swap", a, b)
    def iswap(self, a, b): return self._add("iswap", a, b)
    def crz(self, theta, c, t): return self._add("crz", theta, c, t)
    def cphase(self, theta, c, t): return self._add("cphase", theta, c, t)
    def toffoli(self, c1, c2, t): return self._add("toffoli", c1, c2, t)
    def fredkin(self, c, a, b): return self._add("fredkin", c, a, b)
    def cswap(self, c, a, b): return self._add("fredkin", c, a, b)
    def measure(self, q): return self._add("measure", q)
    def measure_all(self): return self._add("measure_all")

    def __len__(self): return len(self.ops)

    def run(self, shots: int = 1024, seed: Optional[int] = None,
            collapse: bool = False) -> dict[str, int] | Simulator:
        """Execute the circuit. If the last op is a measure, returns a counts
        dict over `shots`. Otherwise returns the final Simulator (no shots used).

        If `collapse=False` (default) and there are no measurement ops, runs
        once and samples from the final state. If there are measurement ops,
        repeats the whole circuit `shots` times for honest sampling.
        """
        has_measure = any(o[0].startswith("measure") for o in self.ops)
        sim = Simulator(self.n_qubits, seed=seed)
        if not has_measure:
            for name, args in self.ops:
                getattr(sim, name)(*args)
            if shots is None:
                return sim
            return sim.sample(shots)

        counts: dict[str, int] = {}
        for _ in range(shots):
            run_sim = Simulator(self.n_qubits, seed=None)
            for name, args in self.ops:
                getattr(run_sim, name)(*args)
            bits = "".join(str(o) for q, o in sorted(run_sim.measurements,
                                                      key=lambda x: -x[0]))
            counts[bits] = counts.get(bits, 0) + 1
        return dict(sorted(counts.items()))
