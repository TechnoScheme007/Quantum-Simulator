"""Quantum Fourier Transform — the workhorse of quantum algorithms.

QFT|x> = (1/sqrt(N)) * sum_y exp(2 pi i x y / N) |y>

For a periodic state, the QFT concentrates probability mass on integer
multiples of N/period — that's the basis of Shor's factoring algorithm.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from qsim import Simulator
from qsim.algorithms import qft
from qsim.visualize import amplitude_table


def main():
    print("=== QFT on each basis state of 3 qubits ===")
    for x in range(8):
        sim = Simulator(3)
        sim.vec = np.zeros(8, dtype=complex)
        sim.vec[x] = 1.0
        qft(sim)
        print(f"QFT|{x}> = {sim.state_str(threshold=1e-3)}")

    print()
    print("=== Periodic state -> QFT concentrates on harmonics ===")
    # Build a 4-periodic state in 5 qubits: |0> + |4> + |8> + |12> + ... + |28>
    n = 5
    sim = Simulator(n)
    sim.vec = np.zeros(2 ** n, dtype=complex)
    for k in range(8):  # 8 copies, spaced by 4
        sim.vec[k * 4] = 1.0
    sim.vec /= np.linalg.norm(sim.vec)
    print(f"Input: periodic state with period 4 ({n} qubits)")
    qft(sim)
    print("After QFT, probability mass concentrates on multiples of 32/4 = 8:")
    probs = sim.probabilities()
    for i, p in enumerate(probs):
        if p > 0.01:
            print(f"  P(|{i:0{n}b}> = {i}) = {p:.4f}")


if __name__ == "__main__":
    main()
