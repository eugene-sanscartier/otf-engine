"""Load the compiled extension that the MTP and the ETN share.

otf_engine._core._core_ext holds the classes. _mtp_ext and _etn_ext import Pair and NeighList from it.

    NeighList      neighbor list that Pair.compute takes
    Pair           base class of PairMTP and PairETN: coefficients, gradients, accessors
    FitStructure   structure to fit, built from an ASE structure in fit_structure.py
    error_table    errors of a potential on structures, as fit_mtp and fit_etn log them
    Equations      MaxVol equations of structures, from the gradients of an MTP or ETN
    MaxVol         active set, to grade against and to search
"""

from importlib import import_module as _import_module

_ext = _import_module("._core_ext", __package__)

NeighList = _ext.NeighList
Pair = _ext.Pair
error_table = _ext.error_table
Equations = _ext.Equations
MaxVol = _ext.MaxVol

__all__ = ["NeighList", "Pair", "error_table", "Equations", "MaxVol"]
