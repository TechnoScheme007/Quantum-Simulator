"""Correctness tests for qsim. Run with: python -m unittest tests.test_qsim"""
import math
import unittest

import numpy as np

from qsim import Simulator, fidelity, basis_state, zero_state
from qsim.gates import (
    I, X, Y, Z, H, S, T, CNOT, CZ, SWAP, TOFFOLI, FREDKIN,
    Rx, Ry, Rz, phase, controlled,
)
from qsim.state import marginal_prob, apply_single, apply_two_qubit, apply_multi
from qsim.algorithms import (
    bell, ghz, teleport, deutsch_jozsa, grover, qft,
    phase_estimation, shor_period, shor_factor,
)


class TestGates(unittest.TestCase):
    def test_pauli_squared_is_identity(self):
        for P in [X, Y, Z]:
            np.testing.assert_allclose(P @ P, np.eye(2), atol=1e-12)

    def test_hadamard_squared_is_identity(self):
        np.testing.assert_allclose(H @ H, np.eye(2), atol=1e-12)

    def test_cnot_self_inverse(self):
        np.testing.assert_allclose(CNOT @ CNOT, np.eye(4), atol=1e-12)

    def test_rotation_at_angle_zero_is_identity(self):
        for R in [Rx, Ry, Rz]:
            np.testing.assert_allclose(R(0), np.eye(2), atol=1e-12)

    def test_rx_at_pi_equals_minus_iX(self):
        np.testing.assert_allclose(Rx(math.pi), -1j * X, atol=1e-12)

    def test_controlled_of_X_equals_CNOT(self):
        np.testing.assert_allclose(controlled(X), CNOT, atol=1e-12)

    def test_toffoli_is_unitary(self):
        np.testing.assert_allclose(TOFFOLI @ TOFFOLI.conj().T,
                                   np.eye(8), atol=1e-12)


class TestStateOps(unittest.TestCase):
    def test_single_qubit_gate_on_correct_axis(self):
        # X on q0 of |00>: should give |01>
        v = zero_state(2)
        v = apply_single(v, X, 0)
        np.testing.assert_allclose(v, basis_state(2, 1), atol=1e-12)

        # X on q1 of |00>: should give |10>
        v = zero_state(2)
        v = apply_single(v, X, 1)
        np.testing.assert_allclose(v, basis_state(2, 2), atol=1e-12)

    def test_cnot_does_textbook_thing(self):
        # CNOT(control=0, target=1) on |01> (q0=1, q1=0) should give |11> (3)
        v = basis_state(2, 1)
        v = apply_two_qubit(v, CNOT, 0, 1)
        np.testing.assert_allclose(v, basis_state(2, 3), atol=1e-12)

    def test_toffoli_only_flips_when_both_controls_one(self):
        """TOFFOLI(c1=q0, c2=q1, target=q2) flips state's q2 iff q0=q1=1.

        In state-index terms (q0 is LSB): when state has q0=1 and q1=1
        (state indices with bits ...11), q2 toggles. That swaps states |3>
        (q2=0,q1=1,q0=1) and |7> (q2=1,q1=1,q0=1)."""
        sim = Simulator(3)
        sim.toffoli(0, 1, 2)  # via Simulator's clean API
        # Identity on |0..0>
        np.testing.assert_allclose(sim.vec, basis_state(3, 0), atol=1e-12)
        # |3> -> |7>
        sim = Simulator(3); sim.vec = basis_state(3, 3)
        sim.toffoli(0, 1, 2)
        np.testing.assert_allclose(sim.vec, basis_state(3, 7), atol=1e-12)
        # |7> -> |3>
        sim = Simulator(3); sim.vec = basis_state(3, 7)
        sim.toffoli(0, 1, 2)
        np.testing.assert_allclose(sim.vec, basis_state(3, 3), atol=1e-12)
        # |6> unchanged (q0=0, no flip)
        sim = Simulator(3); sim.vec = basis_state(3, 6)
        sim.toffoli(0, 1, 2)
        np.testing.assert_allclose(sim.vec, basis_state(3, 6), atol=1e-12)


class TestEntanglement(unittest.TestCase):
    def test_bell_phi_plus(self):
        sim = bell("phi+")
        expected = np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2)
        self.assertGreater(fidelity(sim.vec, expected), 1 - 1e-12)

    def test_bell_psi_minus(self):
        sim = bell("psi-")
        expected = np.array([0, 1, -1, 0], dtype=complex) / math.sqrt(2)
        self.assertGreater(fidelity(sim.vec, expected), 1 - 1e-12)

    def test_ghz_states_are_maximally_entangled(self):
        for n in [2, 3, 4, 5]:
            sim = ghz(n)
            expected = np.zeros(2 ** n, dtype=complex)
            expected[0] = expected[-1] = 1 / math.sqrt(2)
            self.assertGreater(fidelity(sim.vec, expected), 1 - 1e-12)


class TestTeleport(unittest.TestCase):
    def test_teleport_preserves_marginal(self):
        """Teleporting alpha|0>+beta|1> must leave |alpha|^2 on q2 = 0 outcome."""
        for alpha, beta in [
            (1, 0), (0, 1), (0.6, 0.8), (0.6, 0.8j),
            (1 / math.sqrt(2), 1 / math.sqrt(2)),
        ]:
            sim, _ = teleport(alpha, beta, seed=42)
            norm = (abs(alpha) ** 2 + abs(beta) ** 2)
            expected = abs(alpha) ** 2 / norm
            got = marginal_prob(sim.vec, qubit=2, outcome=0)
            self.assertAlmostEqual(got, expected, places=10)


class TestDeutschJozsa(unittest.TestCase):
    def test_constant_zero(self):
        self.assertEqual(deutsch_jozsa(lambda x: 0, n_input=5), "constant")

    def test_constant_one(self):
        self.assertEqual(deutsch_jozsa(lambda x: 1, n_input=5), "constant")

    def test_balanced_parity(self):
        f = lambda x: bin(x).count("1") % 2
        self.assertEqual(deutsch_jozsa(f, n_input=5), "balanced")

    def test_balanced_low_bit(self):
        self.assertEqual(deutsch_jozsa(lambda x: x & 1, n_input=4), "balanced")


class TestGrover(unittest.TestCase):
    def test_single_marked_item_amplified(self):
        n = 5  # 32 entries
        marked = 13
        sim = grover(n, marked=[marked], seed=0)
        probs = sim.probabilities()
        self.assertGreater(probs[marked], 0.95)
        # all others should be small
        for i in range(2 ** n):
            if i != marked:
                self.assertLess(probs[i], 0.01)

    def test_multiple_marked(self):
        n = 6  # 64 entries
        marked = [4, 17, 41]
        sim = grover(n, marked=marked, seed=0)
        probs = sim.probabilities()
        marked_mass = sum(probs[m] for m in marked)
        self.assertGreater(marked_mass, 0.9)


class TestQFT(unittest.TestCase):
    def test_qft_matches_analytic(self):
        """QFT|x> = (1/sqrt(N)) sum_y exp(2 pi i x y / N) |y>"""
        for n in [2, 3, 4]:
            N = 2 ** n
            for x in range(N):
                sim = Simulator(n)
                sim.vec = basis_state(n, x)
                qft(sim)
                expected = np.array(
                    [np.exp(2j * np.pi * x * y / N) for y in range(N)],
                    dtype=complex,
                ) / math.sqrt(N)
                np.testing.assert_allclose(sim.vec, expected, atol=1e-10)

    def test_qft_round_trip_is_identity(self):
        np.random.seed(0)
        for n in [2, 3, 4, 5]:
            v = np.random.randn(2 ** n) + 1j * np.random.randn(2 ** n)
            v /= np.linalg.norm(v)
            sim = Simulator(n)
            sim.vec = v.copy()
            qft(sim)
            qft(sim, inverse=True)
            np.testing.assert_allclose(sim.vec, v, atol=1e-10)


class TestPhaseEstimation(unittest.TestCase):
    def test_exact_phases(self):
        for phi in [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]:
            est = phase_estimation(phi, n_count=5, seed=0)
            self.assertAlmostEqual(est, phi, places=4,
                                   msg=f"phase {phi} estimated as {est}")


class TestShor(unittest.TestCase):
    def test_factor_15(self):
        result = shor_factor(15, attempts=20, seed=42)
        self.assertIsNotNone(result, "Shor should factor 15")
        p, q = result
        self.assertEqual(p * q, 15)
        self.assertIn({p, q}, [{3, 5}])

    def test_factor_21(self):
        result = shor_factor(21, attempts=30, seed=7)
        self.assertIsNotNone(result, "Shor should factor 21")
        p, q = result
        self.assertEqual(p * q, 21)


if __name__ == "__main__":
    unittest.main(verbosity=2)
