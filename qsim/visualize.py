"""
ASCII visualization for quantum states and measurement counts.

No matplotlib — keeps the stack dependency-light. Output is plain text
that renders identically in any terminal.
"""
from __future__ import annotations

import math
import numpy as np


def histogram(counts: dict[str, int], *, width: int = 50, sort: str = "key") -> str:
    """Render a horizontal histogram of measurement counts.

    `counts` is a dict like {"00": 512, "11": 488}.
    `sort` ∈ {"key", "value"}.
    """
    if not counts:
        return "(no counts)"
    total = sum(counts.values())
    items = list(counts.items())
    if sort == "value":
        items.sort(key=lambda x: -x[1])
    else:
        items.sort()
    max_count = max(counts.values())
    out = []
    for label, c in items:
        bar_len = int(round(width * c / max_count))
        bar = "#" * bar_len
        pct = 100 * c / total
        out.append(f"|{label}>  {bar:<{width}}  {c:>6}  {pct:5.1f}%")
    return "\n".join(out)


def amplitude_table(vec: np.ndarray, *, threshold: float = 1e-9) -> str:
    """Tabulate the non-negligible amplitudes and their probabilities."""
    n = int(round(math.log2(vec.size)))
    rows = ["basis     amplitude              probability"]
    rows.append("-" * 50)
    for i, amp in enumerate(vec):
        p = abs(amp) ** 2
        if p < threshold:
            continue
        ket = f"|{i:0{n}b}>"
        a = f"{amp.real:+.4f}{amp.imag:+.4f}j"
        rows.append(f"{ket:<8}  {a:<22} {p:.4f}")
    return "\n".join(rows)


def bloch_ascii(vec: np.ndarray) -> str:
    """Render a single-qubit state as a 2D projection of the Bloch sphere.

    Only works for n=1. For larger states, returns an explanatory message.
    """
    if vec.size != 2:
        return f"(bloch_ascii is single-qubit only; got {vec.size}-dim state)"

    a, b = vec[0], vec[1]
    # Bloch coords: x = 2 Re(a* b), y = 2 Im(a* b), z = |a|^2 - |b|^2
    x = 2 * (a.conjugate() * b).real
    y = 2 * (a.conjugate() * b).imag
    z = abs(a) ** 2 - abs(b) ** 2

    # Draw an XZ-plane projection
    rows, cols = 15, 31
    grid = [[" "] * cols for _ in range(rows)]
    cx, cy = cols // 2, rows // 2

    # Sphere outline
    for theta_deg in range(0, 360, 5):
        th = math.radians(theta_deg)
        gx = int(round(cx + (cols / 2 - 1) * math.cos(th)))
        gy = int(round(cy - (rows / 2 - 1) * math.sin(th)))
        if 0 <= gx < cols and 0 <= gy < rows:
            grid[gy][gx] = "."

    # Axes
    grid[cy][cx] = "+"
    grid[0][cx] = "z"
    grid[rows - 1][cx] = "z"
    grid[cy][0] = "x"
    grid[cy][cols - 1] = "x"

    # State vector projected to (x, z)
    sx = int(round(cx + (cols / 2 - 1) * x))
    sz = int(round(cy - (rows / 2 - 1) * z))
    if 0 <= sx < cols and 0 <= sz < rows:
        grid[sz][sx] = "*"

    out = "\n".join("".join(row) for row in grid)
    out += f"\nBloch coords: x={x:+.3f}  y={y:+.3f}  z={z:+.3f}"
    return out
