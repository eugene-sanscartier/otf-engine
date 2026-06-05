"""Read and write the #MVS_v1.1 active-set section of .almtp/.mtp files.

The section is written by mlip-3's cfg_selection.cpp (Save/Load) and
maxvol.cpp (WriteData/ReadData).  Format (mixed text + binary):

    #MVS_v1.1\\n
    energy_weight <float64>\\n
    force_weight  <float64>\\n
    stress_weight <float64>\\n
    site_en_weight <float64>\\n
    weight_scaling <int>\\n
    #                       <- text '#', immediately followed by binary (no \\n)
    <n*n float64 LE>        <- A matrix, row-major
    <n*n float64 LE>        <- invA matrix, row-major (column-ordered in mlip)
    \\n#\\n                   <- end delimiter

n = CoeffCount() (radial + linear + species coefficients).
n is NOT stored explicitly; it is inferred from the binary block size.
"""

from __future__ import annotations

import math
import struct

import numpy as np

_MARKER = b"#MVS_v1.1"
_END_MARKER = b"\n#\n"


def _find(data: bytes, pattern: bytes, start: int = 0) -> int:
    idx = data.find(pattern, start)
    if idx == -1:
        raise ValueError(f"Pattern {pattern!r} not found in file after offset {start}")
    return idx


def read_active_set(almtp_path: str) -> tuple[dict, np.ndarray, np.ndarray]:
    """Return (weights_dict, A, invA) from the #MVS_v1.1 section.

    A and invA have shape (n, n) where n = CoeffCount() of the potential.
    Raises RuntimeError if the section is absent.
    """
    with open(almtp_path, "rb") as f:
        data = f.read()

    marker_pos = data.find(_MARKER)
    if marker_pos == -1:
        raise RuntimeError(f"No #MVS_v1.1 active-set section found in {almtp_path}.")

    # Parse text weight lines after the marker
    pos = marker_pos + len(_MARKER)
    weights = {}
    # Expect exactly 5 text lines: 4 float weights + 1 int scaling
    for _ in range(5):
        nl = data.find(b"\n", pos)
        line = data[pos:nl].decode().strip()
        pos = nl + 1
        if not line:
            continue
        key, val = line.split(None, 1)
        weights[key] = float(val) if key != "weight_scaling" else int(val)

    # Find the '#' delimiter that immediately precedes the binary data
    hash_pos = data.find(b"#", pos)
    binary_start = hash_pos + 1  # binary follows immediately after '#'

    # Find end delimiter \n#\n
    end_pos = data.find(_END_MARKER, binary_start)
    if end_pos == -1:
        raise ValueError("End delimiter \\n#\\n not found after binary data in #MVS_v1.1 section")

    binary_block = data[binary_start:end_pos]
    n_bytes = len(binary_block)
    # block = A (n*n doubles) + invA (n*n doubles)
    n = int(math.isqrt(n_bytes // 16))
    if 2 * n * n * 8 != n_bytes:
        raise ValueError(f"Binary block size {n_bytes} bytes is not 2*n*n*8 for any integer n")

    A = np.frombuffer(binary_block[:n * n * 8], dtype="<f8").reshape(n, n).copy()
    invA = np.frombuffer(binary_block[n * n * 8:], dtype="<f8").reshape(n, n).copy()
    return weights, A, invA


def write_active_set(almtp_path: str, weights: dict, A: np.ndarray, invA: np.ndarray) -> None:
    """Overwrite the #MVS_v1.1 section in-place.

    Everything before the marker is left unchanged.  If the marker is absent,
    the section is appended at the end of the file.
    """
    with open(almtp_path, "rb") as f:
        data = f.read()

    new_section = _build_section(weights, A, invA)

    marker_pos = data.find(_MARKER)
    if marker_pos == -1:
        # Append
        with open(almtp_path, "ab") as f:
            f.write(b"\n" + new_section)
        return

    # Find end of old section
    binary_start = data.find(b"#", marker_pos + len(_MARKER)) + 1
    end_pos = data.find(_END_MARKER, binary_start)
    if end_pos == -1:
        # Truncate at marker and replace
        with open(almtp_path, "wb") as f:
            f.write(data[:marker_pos])
            f.write(new_section)
        return

    with open(almtp_path, "wb") as f:
        f.write(data[:marker_pos])
        f.write(new_section)
        f.write(data[end_pos + len(_END_MARKER):])


def _build_section(weights: dict, A: np.ndarray, invA: np.ndarray) -> bytes:
    n = A.shape[0]
    lines = [
        b"#MVS_v1.1\n",
        f"energy_weight {weights.get('energy_weight', 1.0):.16e}\n".encode(),
        f"force_weight {weights.get('force_weight', 0.0):.16e}\n".encode(),
        f"stress_weight {weights.get('stress_weight', 0.0):.16e}\n".encode(),
        f"site_en_weight {weights.get('site_en_weight', 1.0):.16e}\n".encode(),
        f"weight_scaling {int(weights.get('weight_scaling', 0))}\n".encode(),
        b"#",  # no newline — binary follows immediately
    ]
    header = b"".join(lines)
    body = A.astype("<f8").tobytes() + invA.astype("<f8").tobytes()
    return header + body + b"\n#\n"
