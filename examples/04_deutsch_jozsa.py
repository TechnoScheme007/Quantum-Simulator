"""Deutsch-Jozsa — distinguish a constant function from a balanced function
with a SINGLE query, when the classical worst case requires 2^(n-1) + 1.

This is the canonical example of quantum speedup, though it only beats
classical determinism — not classical randomness with bounded error.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from qsim.algorithms import deutsch_jozsa


def main():
    n = 6
    test_oracles = {
        "constant f(x) = 0":           lambda x: 0,
        "constant f(x) = 1":           lambda x: 1,
        "balanced (parity of x)":      lambda x: bin(x).count("1") % 2,
        "balanced (high bit of x)":    lambda x: (x >> (n - 1)) & 1,
        "balanced (low bit of x)":     lambda x: x & 1,
    }
    print(f"Deutsch-Jozsa on {n} qubits ({2**n} inputs)")
    print(f"Classical: up to {2**(n-1) + 1} queries needed in worst case")
    print(f"Quantum:   1 query, deterministic answer")
    print()
    print(f"{'oracle':<32}  {'quantum result':<12}")
    print("-" * 50)
    for name, fn in test_oracles.items():
        result = deutsch_jozsa(fn, n_input=n, seed=0)
        print(f"{name:<32}  {result:<12}")


if __name__ == "__main__":
    main()
