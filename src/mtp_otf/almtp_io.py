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

import re
from dataclasses import dataclass
from io import StringIO

import numpy
from numpy import intp, ndarray

_MARKER = b"#MVS_v1.1"
_END_MARKER = b"\n#\n"


@dataclass
class MVSState:
    """Persisted MaxVol state; active_cfg_indices index into selected_cfgs."""

    weights: dict
    A: ndarray
    invA: ndarray
    active_cfg_indices: ndarray
    active_eqn_indices: ndarray
    selected_cfgs: list


def _coeff_count(mtp_text: bytes) -> int:
    fields = {m[1]: int(m[2]) for m in re.finditer(rb"^\s*(\w+)\s*=\s*(\d+)", mtp_text, re.MULTILINE)}
    sc = fields[b"species_count"]
    return fields[b"radial_basis_size"] * sc * sc * fields[b"radial_funcs_count"] + fields[b"alpha_scalar_moments"] + sc


def _read_header_and_binary(data: bytes) -> tuple[dict, ndarray, ndarray, int]:
    marker_pos = data.find(_MARKER)
    if marker_pos == -1:
        raise RuntimeError("No #MVS_v1.1 active-set section found.")

    pos = marker_pos + len(_MARKER)
    weights = {}
    for _ in range(5):
        nl = data.find(b"\n", pos)
        if nl == -1:
            raise ValueError("Unexpected EOF while reading #MVS_v1.1 header")
        line = data[pos:nl].decode().strip()
        pos = nl + 1
        if not line:
            continue
        key, val = line.split(None, 1)
        weights[key] = float(val) if key != "weight_scaling" else int(val)

    binary_start = data.find(b"#", pos)
    if binary_start == -1:
        raise ValueError("Binary payload marker '#' not found in #MVS_v1.1 section")
    binary_start += 1
    n = _coeff_count(data[:marker_pos])
    n_bytes = 2 * n * n * 8
    end_pos = binary_start + n_bytes
    if data[end_pos:end_pos + len(_END_MARKER)] != _END_MARKER:
        raise ValueError(f"Expected '\\n#\\n' at binary end position {end_pos}, got {data[end_pos:end_pos+3]!r}")

    A = numpy.frombuffer(data[binary_start:binary_start + n * n * 8], dtype="<f8").reshape(n, n).copy()
    invA = numpy.frombuffer(data[binary_start + n * n * 8:end_pos], dtype="<f8").reshape(n, n).copy()
    return weights, A, invA, end_pos


def _read_links_and_cfgs(data: bytes, n: int, end_pos: int) -> tuple[ndarray, ndarray, list]:
    from .io_cfg import read_cfg

    pos = end_pos + len(_END_MARKER)
    active_cfg_indices = numpy.full(n, -1, dtype=intp)
    active_eqn_indices = numpy.full(n, -1, dtype=intp)
    for i in range(n):
        nl = data.find(b"\n", pos)
        if nl == -1:
            raise ValueError("Unexpected EOF while reading active-set link table")
        line = data[pos:nl].decode().strip()
        pos = nl + 1
        if not line:
            continue
        cfg_idx, eqn_idx = line.split(None, 1)
        active_cfg_indices[i] = int(cfg_idx)
        active_eqn_indices[i] = int(eqn_idx)

    if data[pos:pos + 2] != b"#\n":
        raise ValueError("Active-set link terminator '#\\n' not found after link table")

    cfg_text = data[pos + 2:].decode()
    selected_cfgs = [] if not cfg_text.strip() else read_cfg(StringIO(cfg_text))
    return active_cfg_indices, active_eqn_indices, selected_cfgs


def read_mvs_state(almtp_path: str) -> MVSState:
    """Return the full mlip-3 active-learning footer state."""
    with open(almtp_path, "rb") as f:
        data = f.read()

    try:
        weights, A, invA, end_pos = _read_header_and_binary(data)
    except RuntimeError as exc:
        raise RuntimeError(f"{exc.args[0]} in {almtp_path}.") from exc

    active_cfg_indices, active_eqn_indices, selected_cfgs = _read_links_and_cfgs(data, A.shape[0], end_pos)
    return MVSState(weights=weights, A=A, invA=invA, active_cfg_indices=active_cfg_indices, active_eqn_indices=active_eqn_indices, selected_cfgs=selected_cfgs)


def _encode_mvs_state(state: MVSState) -> bytes:
    from .io_cfg import write_cfg

    n = state.A.shape[0]
    if state.invA.shape != (n, n):
        raise ValueError(f"invA shape {state.invA.shape} does not match A shape {state.A.shape}")

    active_cfg_indices = numpy.asarray(state.active_cfg_indices, dtype=intp)
    active_eqn_indices = numpy.asarray(state.active_eqn_indices, dtype=intp)
    if active_cfg_indices.shape != (n, ):
        raise ValueError(f"active_cfg_indices shape {active_cfg_indices.shape} does not match ({n},)")
    if active_eqn_indices.shape != (n, ):
        raise ValueError(f"active_eqn_indices shape {active_eqn_indices.shape} does not match ({n},)")

    section = b"".join((
        b"#MVS_v1.1\n",
        f"energy_weight {state.weights.get('energy_weight', 1.0):.16e}\n".encode(),
        f"force_weight {state.weights.get('force_weight', 0.0):.16e}\n".encode(),
        f"stress_weight {state.weights.get('stress_weight', 0.0):.16e}\n".encode(),
        f"site_en_weight {state.weights.get('site_en_weight', 1.0):.16e}\n".encode(),
        f"weight_scaling {int(state.weights.get('weight_scaling', 0))}\n".encode(),
        b"#",
        state.A.astype("<f8").tobytes(),
        state.invA.astype("<f8").tobytes(),
        b"\n#\n",
    ))

    link_lines = [f"{int(cfg_idx)} {int(eqn_idx)}\n".encode() for cfg_idx, eqn_idx in zip(active_cfg_indices, active_eqn_indices, strict=True)]
    section += b"".join(link_lines) + b"#\n"

    if state.selected_cfgs:
        sio = StringIO()
        write_cfg(sio, state.selected_cfgs)
        section += sio.getvalue().encode()

    return section


def write_mvs_state(almtp_path: str, state: MVSState) -> None:
    """Write the full mlip-3 active-learning footer state."""
    new_section = _encode_mvs_state(state)

    with open(almtp_path, "rb") as f:
        data = f.read()

    marker_pos = data.find(_MARKER)
    if marker_pos == -1:
        with open(almtp_path, "ab") as f:
            f.write(b"\n" + new_section)
        return

    with open(almtp_path, "wb") as f:
        f.write(data[:marker_pos])
        f.write(new_section)
