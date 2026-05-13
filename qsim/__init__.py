"""qsim — a state-vector quantum simulator with the canonical algorithms.

Public API:

    from qsim import Simulator, Circuit
    from qsim.gates import H, X, Y, Z, CNOT, ...
    from qsim.algorithms import bell, grover, qft, teleport, ...
"""
from .simulator import Simulator, Circuit
from .state import (
    zero_state, basis_state, probabilities, marginal_prob,
    project_measure, state_to_str, fidelity,
)

__version__ = "0.1.0"
__all__ = [
    "Simulator", "Circuit",
    "zero_state", "basis_state", "probabilities", "marginal_prob",
    "project_measure", "state_to_str", "fidelity",
]
