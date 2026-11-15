"""Source wrapper for the compiled ETN extension.

The binary implementation lives in otf_engine._etn._etn_ext, whose classes derive
from _core_ext's Pair.

    PairETN          energy, forces, virial, and each atom's input and bond vectors (a Pair)
    ETNTraining      + derivatives of site energies and of a loss w.r.t. the coefficients
    train_etn        fits an ETNTraining with ETNTrainer
    real_harmonics   the real spherical harmonics the input vector is built on
    wigner_3j        3-j symbols of complex harmonics
    real_wigner_3j   3-j symbols of real harmonics, which the cores carry
"""

from importlib import import_module as _import_module

_ext = _import_module("._etn_ext", __package__)

PairETN = _ext.PairETN
ETNTraining = _ext.ETNTraining
train_etn = _ext.train_etn
real_harmonics = _ext.real_harmonics
wigner_3j = _ext.wigner_3j
real_wigner_3j = _ext.real_wigner_3j

__all__ = ["PairETN", "ETNTraining", "train_etn", "real_harmonics", "wigner_3j", "real_wigner_3j"]
