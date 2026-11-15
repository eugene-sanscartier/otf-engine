"""Source wrapper for the compiled ETN extension.

The binary implementation lives in otf_engine._etn._etn_ext, whose compute takes
_mtp_ext's NeighList; importing otf_engine._mtp first registers that type.

    PairETN          energy, forces, virial, and each atom's input and bond vectors
    ETNTraining      + derivatives of site energies and of a loss w.r.t. the coefficients
    real_harmonics   the real spherical harmonics the input vector is built on
    wigner_3j        3-j symbols of complex harmonics
    real_wigner_3j   3-j symbols of real harmonics, which the cores carry
"""

from importlib import import_module as _import_module

from .. import _mtp  # noqa: F401  registers NeighList

_ext = _import_module("._etn_ext", __package__)

PairETN = _ext.PairETN
ETNTraining = _ext.ETNTraining
real_harmonics = _ext.real_harmonics
wigner_3j = _ext.wigner_3j
real_wigner_3j = _ext.real_wigner_3j

__all__ = ["PairETN", "ETNTraining", "real_harmonics", "wigner_3j", "real_wigner_3j"]
