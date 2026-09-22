"""Neighbor lists for the MTP kernels."""

import numpy
from numpy import int32, float64, ndarray
from ase.neighborlist import neighbor_list

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
    """
    if types is None:
        types = mtp_types(atoms)

    i_arr, j_arr, D_arr = neighbor_list("ijD", atoms, cutoff)

    n_atoms = len(atoms)
    ilist = numpy.arange(n_atoms, dtype=int32)
    numneigh = numpy.bincount(i_arr, minlength=n_atoms).astype(int32)

    # Sort so neighbor blocks are contiguous for each central atom
    order = numpy.argsort(i_arr, kind="stable")
    firstneigh = j_arr[order].astype(int32)
    displacements = numpy.ascontiguousarray(D_arr[order], dtype=float64)

    return NeighList(types, ilist, numneigh, firstneigh, displacements)
