"""Neighbor lists for the MTP kernels."""

import itertools

import numpy
from numpy import int32, ndarray
from scipy.spatial import cKDTree

from ._mtp import NeighList


def mtp_types(atoms) -> ndarray:
    """Return the 0-indexed MTP species array for *atoms*.

    Uses ``atoms.arrays["type_index"]`` when present; otherwise assigns indices
    by order of first appearance of chemical symbols.
    """
    if "type_index" in atoms.arrays:
        return numpy.asarray(atoms.arrays["type_index"], dtype=int32)

    seen: dict[str, int] = {}
    for sym in atoms.get_chemical_symbols():
        if sym not in seen:
            seen[sym] = len(seen)
    return numpy.array([seen[s] for s in atoms.get_chemical_symbols()], dtype=int32)


def neighbors(atoms, cutoff: float, types: ndarray | None = None) -> NeighList:
    """Build the MTP neighbor list for *atoms* within *cutoff*.

    Every pair is listed in both directions — equivalent to LAMMPS REQ_FULL —
    and the displacements carry the PBC shift, ``D = r_j - r_i + S @ cell``.
    Any cell shape works, including triclinic cells and cells shorter than *cutoff*.
    """
    if types is None:
        types = mtp_types(atoms)

    n_atoms = len(atoms)
    inv_cell = numpy.linalg.inv(atoms.cell.array)
    frac = atoms.positions @ inv_cell
    frac[:, atoms.pbc] %= 1.0

    # Periodic images within the cutoff of the cell; image k is a copy of atom k % n_atoms.
    pad = cutoff * numpy.linalg.norm(inv_cell, axis=0)
    reps = numpy.ceil(pad).astype(int) * atoms.pbc
    shifts = numpy.array(list(itertools.product(*[range(-n, n + 1) for n in reps])))
    image_frac = (shifts[:, None, :] + frac[None, :, :]).reshape(-1, 3)
    image_ids = numpy.flatnonzero(numpy.all((image_frac > -pad) & (image_frac < 1 + pad) | ~atoms.pbc, axis=1))

    positions = frac @ atoms.cell.array
    images = image_frac[image_ids] @ atoms.cell.array
    pairs = cKDTree(positions).sparse_distance_matrix(cKDTree(images), cutoff, output_type="ndarray")
    pairs = pairs[(pairs["v"] > 0.0) & (pairs["v"] < cutoff)]
    order = numpy.argsort(pairs["i"], kind="stable")
    i, j = pairs["i"][order], pairs["j"][order]

    numneigh = numpy.bincount(i, minlength=n_atoms).astype(int32)
    firstneigh = (image_ids[j] % n_atoms).astype(int32)
    return NeighList(types, numpy.arange(n_atoms, dtype=int32), numneigh, firstneigh, images[j] - positions[i])
