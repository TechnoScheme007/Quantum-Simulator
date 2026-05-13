"""Shor's algorithm — factor a composite integer using quantum period finding.

The classical part: given a, find the period r of a^x mod N.
The quantum part: estimate r efficiently via Quantum Phase Estimation on
the modular-multiplication unitary U|y> = |a*y mod N>.

This simulation factors small N (up to ~21) on a laptop. The full simulator
state is 2^(n_count + n_work) amplitudes — for N=15, that's 2^12 = 4096
amplitudes. For N=2048-bit RSA, it would be 2^(2*2048 + 2048) = an
astronomical 2^6144 amplitudes. The quantum speedup is real; the simulation
cost grows exponentially with N, which is why nobody has factored real RSA
keys yet.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import math
from qsim.algorithms import shor_factor, shor_period


def main():
    print("=== Period finding (the quantum subroutine) ===")
    print()
    cases = [(7, 15, 4), (2, 15, 4), (4, 15, 2), (2, 21, 6)]
    print(f"{'a':>3}  {'N':>3}  {'true r':>8}  {'measured r':>11}")
    print("-" * 35)
    for a, N, true_r in cases:
        # try a few seeds and pick the right answer
        results = [shor_period(a, N, n_count=2 * math.ceil(math.log2(N)),
                                seed=s) for s in range(8)]
        good = [r for r in results if r == true_r]
        print(f"{a:>3}  {N:>3}  {true_r:>8}  {results} "
              f"({len(good)}/{len(results)} correct)")

    print()
    print("=== Factoring small composites ===")
    for N in [15, 21, 33, 35]:
        result = shor_factor(N, attempts=20, seed=N)
        if result:
            p, q = result
            print(f"  {N} = {p} x {q}  [verified: {p*q == N}]")
        else:
            print(f"  {N}: failed after 20 attempts")


if __name__ == "__main__":
    main()
