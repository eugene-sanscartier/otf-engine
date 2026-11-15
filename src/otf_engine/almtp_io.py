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
n is NOT stored explicitly; it is inferred from the binary block size.
"""

from __future__ import annotations

from io import StringIO

import numpy
from numpy import intp, ndarray

from ._core import MaxVol

_MARKER = b"#MVS_v1.1"
_END_MARKER = b"\n#\n"


def _read_header_and_binary(data: bytes, path: str) -> tuple[dict, ndarray, ndarray, int]:
    marker_pos = data.find(_MARKER)
    if marker_pos == -1:
        raise RuntimeError("No #MVS_v1.1 active-set section found.")

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
    from .models import model_of    # loads the extensions, which reading the section's header does not need
    n = model_of(path).training(path).get_coeff_count()
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


def read_mvs_header(almtp_path: str) -> tuple[dict, MaxVol]:
    """The selection weights and the active set stored in *almtp_path*, without the stored structures or the rows' provenance."""
    with open(almtp_path, "rb") as f:
        data = f.read()

    try:
        weights, A, invA, _ = _read_header_and_binary(data, almtp_path)
    except RuntimeError as exc:
        raise RuntimeError(f"{exc.args[0]} in {almtp_path}.") from exc
    return weights, MaxVol(A, invA)


def read_mvs_state(almtp_path: str, pool_id: int = 0) -> tuple[dict, MaxVol, list]:
    """The selection weights, the active set and the stored structures of *almtp_path*.

    An active row linked to a stored structure gets *pool_id* as its pool and the structure's index in the returned list.
    """
    with open(almtp_path, "rb") as f:
        data = f.read()

    try:
        weights, A, invA, end_pos = _read_header_and_binary(data, almtp_path)
    except RuntimeError as exc:
        raise RuntimeError(f"{exc.args[0]} in {almtp_path}.") from exc

    active_cfg_indices, active_eqn_indices, selected_cfgs = _read_links_and_cfgs(data, A.shape[0], end_pos)
    active_set = MaxVol(A, invA)
    active_set.restore_active(active_cfg_indices, active_eqn_indices, pool_id)
    return weights, active_set, selected_cfgs


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
        f"energy_weight {weights.get('energy_weight', 1.0):.16e}\n".encode(),
        f"force_weight {weights.get('force_weight', 0.0):.16e}\n".encode(),
        f"stress_weight {weights.get('stress_weight', 0.0):.16e}\n".encode(),
        f"site_en_weight {weights.get('site_en_weight', 1.0):.16e}\n".encode(),
        f"weight_scaling {int(weights.get('weight_scaling', 0))}\n".encode(),
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
