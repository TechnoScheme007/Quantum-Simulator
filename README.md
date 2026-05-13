# qsim

A from-scratch state-vector quantum computing simulator and a small library of canonical quantum algorithms. Pure Python on top of NumPy. No `qiskit`, no `cirq`, no `pyquil` — every gate, every algorithm, every line of math is here in the repo for you to read.

## What this is

The **lowest layer of a quantum software stack**: the part that takes a circuit description and produces correct quantum-mechanical behavior. The same role Qiskit's `statevector_simulator` plays, but small enough that you can read it end to end in an evening.

It is **not** a full production quantum stack. It does not have an optimizing compiler that routes qubits onto a real hardware topology, surface-code error correction, OpenPulse hardware control, a high-level programming language with entanglement-tracking type system, or a chemistry frontend that solves ground states of 50-electron molecules. Those are real research-grade systems and they take teams years. This is the foundation that the rest of that stack would sit on.

## What it can do, demonstrably

All of the following are verified by the test suite and reproduced by the example scripts:

- **Bell and GHZ states** — entanglement with machine-precision fidelity to the analytical states
- **Quantum teleportation** — the marginal probability on the receiving qubit equals `|alpha|^2` for any input state `alpha|0> + beta|1>`, across random measurement outcomes
- **Deutsch-Jozsa** — distinguishes constant from balanced functions on n qubits with a single quantum query
- **Grover's search** — locates a marked needle in `N = 2^n` items in `O(sqrt(N))` iterations with > 99% probability for single marked items
- **Quantum Fourier Transform** — matches `(1/sqrt(N)) * sum_y exp(2*pi*i*x*y/N) |y>` for every `(n, x)` to machine precision, with proper round-trip inverse
- **Quantum Phase Estimation** — recovers exact phases at the resolvable grid (`1/2^n_count`)
- **Shor's algorithm** — factors `N = 15, 21, 33, 35` via real quantum period finding (QPE on modular-multiplication unitary)

## Install and run

```bash
python -m venv .venv && source .venv/bin/activate    # or just use system python
pip install -r requirements.txt
python -m unittest tests.test_qsim                    # 25 tests, ~0.1s
python examples/01_bell_state.py                      # try any example
```

Each example is a standalone script. They run from any directory; the path bootstrap at the top of each file makes them work without installing the package.

## Library at a glance

```python
from qsim import Simulator
from qsim.algorithms import bell, grover, qft, teleport, shor_factor

# Imperative style
sim = Simulator(2, seed=42)
sim.h(0).cnot(0, 1)
print(sim.sample(shots=4096))        # {'00': 2074, '11': 2022}

# Algorithms as building blocks
sim = grover(n=6, marked=[42], seed=0)
print(sim.probabilities()[42])       # 0.9966

# Shor's algorithm — real quantum factoring
print(shor_factor(15))               # (3, 5)
```

## Repository layout

```
quantum computing/
├── qsim/
│   ├── state.py          # Quantum state + tensor-based gate application
│   ├── gates.py          # Standard gate library (Pauli, H, S, T, R*, CNOT, ...)
│   ├── simulator.py      # Simulator class + declarative Circuit DSL
│   ├── algorithms.py     # Bell, GHZ, teleport, DJ, Grover, QFT, QPE, Shor
│   └── visualize.py      # ASCII histograms + amplitude tables
├── examples/
│   ├── 01_bell_state.py
│   ├── 02_grover.py
│   ├── 03_teleportation.py
│   ├── 04_deutsch_jozsa.py
│   ├── 05_qft.py
│   └── 06_shor.py
├── tests/
│   └── test_qsim.py      # 25 tests covering gates, algorithms, edge cases
├── requirements.txt
└── README.md
```

## Conventions worth knowing

- **Qubit ordering**: qubit 0 is the least-significant bit, matching Qiskit. So `|5>` in a 3-qubit system means `q2=1, q1=0, q0=1`.
- **Two-qubit gates**: a 4x4 gate matrix is in the `|q_a q_b>` basis where `q_a` is the high bit. Pass control as `q_a` and target as `q_b` and the textbook CNOT matrix does what you expect.
- **State application**: gates are applied via NumPy `reshape` + `moveaxis` rather than building the full `2^n` by `2^n` unitary. Memory cost is `O(2^n)`, not `O(4^n)`, which is why simulating up to ~25 qubits is tractable on a laptop.

## Honest limits

- **Scale**: state-vector simulation of `n` qubits requires `2^n` complex amplitudes = `16 * 2^n` bytes. 20 qubits ≈ 16 MB. 25 qubits ≈ 512 MB. 30 qubits ≈ 16 GB. Above that, this approach doesn't fit on a laptop. Tensor-network and stabilizer simulators handle special structured states up to thousands of qubits; this implementation does not.
- **Shor's algorithm**: the simulation works on small `N` (up to about 33 on a 16 GB laptop, since it needs `2*ceil(log2 N)` counting qubits plus `ceil(log2 N)` work qubits). The quantum speedup is real — but classically *simulating* a quantum computer scales exponentially, which is why nobody has used a simulator to factor real RSA keys.
- **No noise model**: the simulator is ideal. There is no decoherence, no gate error, no measurement noise. Real quantum hardware has all three, and a production stack would model them.
- **No optimizer**: the gate sequences are executed as written. There is no peephole optimization, gate fusion, or qubit-routing pass.

If any of these limits matter to you, this is the part of the stack you would extend.

## References

- Nielsen and Chuang, *Quantum Computation and Quantum Information*
- Mermin, *Quantum Computer Science: An Introduction*
- Original papers: Deutsch 1985, Bennett et al. 1993, Grover 1996, Shor 1994

## License

MIT.
