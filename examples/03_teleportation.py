"""Quantum teleportation — transfer an unknown qubit state between two parties
using one shared Bell pair and two classical bits.

Demonstrates the key fact: NO physical particle moves from Alice to Bob.
Just two classical bits — but those bits, combined with their pre-shared
entanglement, reconstruct the exact quantum state on Bob's side.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from qsim.algorithms import teleport
from qsim.state import marginal_prob


def main():
    print("Teleporting state alpha|0> + beta|1>  from q0 to q2")
    print()
    test_states = [
        (1, 0),                          # |0>
        (0, 1),                          # |1>
        (1/np.sqrt(2), 1/np.sqrt(2)),    # |+>
        (1/np.sqrt(2), -1/np.sqrt(2)),   # |->
        (0.6, 0.8),                      # arbitrary
        (0.6, 0.8j),                     # arbitrary complex
    ]
    print(f"{'alpha':>20}  {'beta':>20}  {'|alpha|^2':>10}  {'P(q2=0)':>9}  {'OK?'}")
    print("-" * 80)
    for alpha, beta in test_states:
        norm = (abs(alpha)**2 + abs(beta)**2) ** 0.5
        a, b = alpha / norm, beta / norm
        sim, (m0, m1) = teleport(alpha, beta, seed=hash((alpha, beta)) & 0xFFFF)
        p0 = marginal_prob(sim.vec, qubit=2, outcome=0)
        expected_p0 = abs(a) ** 2
        ok = "OK" if abs(p0 - expected_p0) < 1e-10 else "FAIL"
        print(f"{str(a):>20}  {str(b):>20}  {expected_p0:10.4f}  {p0:9.4f}  {ok}")
    print()
    print("The classical bits Alice sends (m0, m1) are different every run,")
    print("but P(q2=0) always equals |alpha|^2 — the original state is intact.")


if __name__ == "__main__":
    main()
