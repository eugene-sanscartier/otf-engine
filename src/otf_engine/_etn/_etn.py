"""Source wrapper for the compiled ETN extension.

The binary implementation lives in otf_engine._etn._etn_ext, whose classes derive
from _core_ext's Pair.

    PairETN          energy, forces, virial, and each atom's input and bond vectors (a Pair)
    ETN              + derivatives of site energies and of a loss w.r.t. the coefficients
    fit_etn        fits an ETN with ETNFit
    real_harmonics   the real spherical harmonics the input vector is built on
    wigner_3j        3-j symbols of complex harmonics
    real_wigner_3j   3-j symbols of real harmonics, which the cores carry

PairETN and ETN load an .etn file or build the untrained ETN of a shape given by the file's fields.
PairETN carries what the selection and the training call for every ETN: write, fit (fit_etn),
and mlip-4's CfgCalc rules threshold_swap and retrain_pass.
"""

from importlib import import_module as _import_module

from .io import write_etn

_ext = _import_module("._etn_ext", __package__)

PairETN = _ext.PairETN
ETN = _ext.ETN
fit_etn = _ext.fit_etn
real_harmonics = _ext.real_harmonics
wigner_3j = _ext.wigner_3j
real_wigner_3j = _ext.real_wigner_3j

PairETN.write = write_etn                                  # potential.write(path)
PairETN.fit = staticmethod(fit_etn)
PairETN.threshold_swap = 1.00001    # mlip-4's MaximizeVol goes on swapping, once a search has swapped, while a grade exceeds this
PairETN.retrain_pass = False        # mlip-4's CfgCalc makes no training pass after the candidates

__all__ = ["PairETN", "ETN", "fit_etn", "real_harmonics", "wigner_3j", "real_wigner_3j"]
