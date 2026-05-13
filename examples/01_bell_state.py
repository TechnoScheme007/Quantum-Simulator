"""Bell state preparation — the simplest example of entanglement.

|phi+> = (|00> + |11>) / sqrt(2)

Measuring either qubit alone is 50/50 random. Measuring both reveals they
are perfectly correlated — never |01> or |10>.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from qsim import Simulator
from qsim.algorithms import bell
from qsim.visualize import histogram, amplitude_table


def main():
    print("Preparing Bell state |phi+> = (|00> + |11>)/sqrt(2)")
    print()
    sim = bell("phi+", seed=42)
    print(amplitude_table(sim.vec))
    print()
    print("Sampling 4096 shots:")
    print(histogram(sim.sample(shots=4096)))


if __name__ == "__main__":
    main()
