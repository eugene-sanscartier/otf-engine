"""Load the compiled ETN extension.

otf_engine._etn._etn_ext holds the classes, which derive from the Pair of _core_ext.

    PairETN               energy, forces, virial (a Pair)
    PairETNExtrapolation  + extrapolation grades against an active set (a PairETN)
    ETN                   + input and bond vectors of each atom, derivatives of the site
                            energies and of a loss w.r.t. the coefficients
    fit_etn               fit an ETN with ETNFit
    real_harmonics        real spherical harmonics that build the input vector
    wigner_3j             3-j symbols of complex harmonics
    real_wigner_3j        3-j symbols of real harmonics, which the cores carry

PairETN and ETN load an .etn file, or build an untrained ETN from the fields of the file.
PairETN carries the write and fit (fit_etn) methods that the training calls.
"""

from importlib import import_module as _import_module

from .io import write_etn

_ext = _import_module("._etn_ext", __package__)

PairETN = _ext.PairETN
PairETNExtrapolation = _ext.PairETNExtrapolation
ETN = _ext.ETN
fit_etn = _ext.fit_etn
real_harmonics = _ext.real_harmonics
wigner_3j = _ext.wigner_3j
real_wigner_3j = _ext.real_wigner_3j

PairETN.write = write_etn
PairETN.fit = staticmethod(fit_etn)

__all__ = ["PairETN", "PairETNExtrapolation", "ETN", "fit_etn", "real_harmonics", "wigner_3j", "real_wigner_3j"]
