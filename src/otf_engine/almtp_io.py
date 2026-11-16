"""Read and write the #MVS_v1.1 active-set section of .almtp/.mtp and .etn files.

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

n = CoeffCount() (radial + linear + species coefficients), or an ETN's coefficient count.
The section does not store n; the reader takes it from the potential in the same file.
"""

from __future__ import annotations

from io import StringIO

import numpy
from numpy import intp, ndarray

from ._core import MaxVol, read_potential

_MARKER = b"#MVS_v1.1"
_END_MARKER = b"\n#\n"


def _read_section(path: str) -> tuple[bytes, dict, ndarray, ndarray, int]:
    """The bytes of the file *path*, and the weights, A, invA and the end of the binary block of its #MVS_v1.1 section."""
    with open(path, "rb") as f:
        data = f.read()

    marker_pos = data.find(_MARKER)
    if marker_pos == -1:
        raise RuntimeError(f"No #MVS_v1.1 active-set section in {path}.")

    pos = marker_pos + len(_MARKER)
    weights = {}
    while len(weights) < 5:
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
    n = read_potential(path).get_coeff_count()
    n_bytes = 2 * n * n * 8
    end_pos = binary_start + n_bytes
    if data[end_pos:end_pos + len(_END_MARKER)] != _END_MARKER:
        raise ValueError(f"Expected '\\n#\\n' at binary end position {end_pos}, got {data[end_pos:end_pos+3]!r}")

    A = numpy.frombuffer(data[binary_start:binary_start + n * n * 8], dtype="<f8").reshape(n, n).copy()
    invA = numpy.frombuffer(data[binary_start + n * n * 8:end_pos], dtype="<f8").reshape(n, n).copy()
    return data, weights, A, invA, end_pos


def read_mvs_header(almtp_path: str) -> tuple[dict, MaxVol]:
    """The selection weights and the active set stored in *almtp_path*, without the stored structures or the rows' provenance."""
    _, weights, A, invA, _ = _read_section(almtp_path)
    return weights, MaxVol(A, invA)


def read_mvs_state(almtp_path: str, pool_id: int = 0) -> tuple[dict, MaxVol, list]:
    """The selection weights, the active set and the stored structures of *almtp_path*.

    An active row linked to a stored structure gets *pool_id* as its pool and the structure's index in the returned list.
    """
    from .io_cfg import read_cfg

    data, weights, A, invA, end_pos = _read_section(almtp_path)

    # The link table, a structure index and an equation index per row of A, then the stored structures
    n = A.shape[0]
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
    stored_structs = [] if not cfg_text.strip() else read_cfg(StringIO(cfg_text))

    active_set = MaxVol(A, invA)
    active_set.restore_active(active_cfg_indices, active_eqn_indices, pool_id)
    return weights, active_set, stored_structs


def write_mvs_state(almtp_path: str, weights: dict, active_set: MaxVol, structures: list, pool_id: int) -> list:
    """Write *weights* and *active_set* as the #MVS_v1.1 section of *almtp_path*, with the structures of *structures*
    that the rows of pool *pool_id* come from. Returns those structures."""
    from .io_cfg import write_cfg

    # The structures rows of pool_id come from, each with the indices of its rows' equations, and each row's index among them
    rows = [(int(p), int(s), int(e)) for p, s, e in zip(active_set.active_pool_ids, active_set.active_struct_indices, active_set.active_eqn_indices, strict=True)]
    labels = sorted({s for p, s, _ in rows if p == pool_id and 0 <= s < len(structures)})
    stored = []
    for label in labels:
        atoms = structures[label].copy()
        atoms.calc = structures[label].calc
        eqn_indices = sorted({e for p, s, e in rows if p == pool_id and s == label and e >= 0})
        if eqn_indices:
            atoms.info.setdefault("features", {})["selected_eqn_inds"] = ",".join(str(e) for e in eqn_indices)
        stored += [atoms]
    index_of = {label: i for i, label in enumerate(labels)}
    cfg_indices = [index_of.get(s, -1) if p == pool_id else -1 for p, s, _ in rows]

    section = b"".join((
        b"#MVS_v1.1\n",
        f"energy_weight {weights['energy_weight']:.16e}\n".encode(),
        f"force_weight {weights['force_weight']:.16e}\n".encode(),
        f"stress_weight {weights['stress_weight']:.16e}\n".encode(),
        f"site_en_weight {weights['site_en_weight']:.16e}\n".encode(),
        f"weight_scaling {int(weights['weight_scaling'])}\n".encode(),
        b"#",
        numpy.asarray(active_set.A).astype("<f8").tobytes(),
        numpy.asarray(active_set.invA).astype("<f8").tobytes(),
        b"\n#\n",
    ))
    section += b"".join(f"{int(c)} {int(e)}\n".encode() for c, e in zip(cfg_indices, active_set.active_eqn_indices, strict=True)) + b"#\n"
    if stored:
        sio = StringIO()
        write_cfg(sio, stored)
        section += sio.getvalue().encode()

    with open(almtp_path, "rb") as f:
        data = f.read()
    marker_pos = data.find(_MARKER)
    if marker_pos == -1:
        with open(almtp_path, "ab") as f:
            f.write(b"\n" + section)
    else:
        with open(almtp_path, "wb") as f:
            f.write(data[:marker_pos] + section)
    return stored
