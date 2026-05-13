"""Grover's search — find a marked needle in an unstructured haystack of N=2^n
entries, using ~sqrt(N) quantum queries instead of N/2 classical ones.

This demo searches a 6-qubit space (N=64) for the marked entry x=42, using
the optimal number of iterations: floor(pi/4 * sqrt(64)) = 6.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import math
from qsim.algorithms import grover
from qsim.visualize import histogram


def main():
    n_qubits = 6
    target = 42
    print(f"Searching {2**n_qubits} entries for x = {target}")
    print(f"Classical worst case: {2**n_qubits} queries")
    print(f"Classical expected:   {2**n_qubits // 2} queries")
    iters = max(1, int(round((math.pi / 4) * math.sqrt(2 ** n_qubits))))
    print(f"Grover:               {iters} oracle calls")
    print()
    sim = grover(n_qubits, marked=[target], seed=42)
    probs = sim.probabilities()
    print(f"P(measure {target:>2}) = {probs[target]:.4f}  "
          f"(uniform would be {1/2**n_qubits:.4f})")
    print()
    counts = sim.sample(shots=2048)
    # show only the top 5
    top = dict(sorted(counts.items(), key=lambda x: -x[1])[:5])
    print("Top-5 measured bitstrings (2048 shots):")
    print(histogram(top, width=40, sort="value"))


if __name__ == "__main__":
    main()
