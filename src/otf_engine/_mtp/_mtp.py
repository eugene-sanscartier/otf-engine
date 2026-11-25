"""Load the compiled MTP extension.

otf_engine._mtp._mtp_ext holds the classes, which follow the C++ hierarchy on the Pair of _core:
    PairMTP                 energy, forces, virial       (a Pair)
    PairMTPExtrapolation    + extrapolation grades       (a PairMTP)
    MTP                     + basis values and derivatives
                              w.r.t. the coefficients    (a PairMTP)

Each class loads a file, or builds the untrained MTP of a level or of basis functions.
level_basis_functions returns the basis functions of a level. fit_mtp fits an MTP with MTPFit.
PairMTP carries the write and fit (fit_mtp) methods that the training calls.
"""

from importlib import import_module as _import_module

from .io import write_mtp

_ext = _import_module("._mtp_ext", __package__)

PairMTP = _ext.PairMTP
PairMTPExtrapolation = _ext.PairMTPExtrapolation
MTP = _ext.MTP
fit_mtp = _ext.fit_mtp
level_basis_functions = _ext.level_basis_functions

PairMTP.write = write_mtp
PairMTP.fit = staticmethod(fit_mtp)

__all__ = ["PairMTP", "PairMTPExtrapolation", "MTP", "fit_mtp", "level_basis_functions"]
