"""Source wrapper for the compiled MTP extension.

The binary implementation lives in otf_engine._mtp._mtp_ext. This wrapper keeps
the public import path source-backed so Pylance can resolve it without
suppressing diagnostics.

The three classes mirror the C++ hierarchy, on _core's Pair:
    PairMTP                 energy, forces, virial       (a Pair)
    PairMTPExtrapolation    + extrapolation grades       (a PairMTP)
    MTP                     + basis values and derivatives
                              w.r.t. the coefficients    (a PairMTP)

Each class loads a file or builds the untrained MTP of a level or of basis functions;
level_basis_functions gives a level's. fit_mtp fits an MTP with MTPFit. PairMTP carries what the training calls for
every MTP: write and fit (fit_mtp).
"""

from importlib import import_module as _import_module

from .io import write_mtp

_ext = _import_module("._mtp_ext", __package__)

PairMTP = _ext.PairMTP
PairMTPExtrapolation = _ext.PairMTPExtrapolation
MTP = _ext.MTP
fit_mtp = _ext.fit_mtp
level_basis_functions = _ext.level_basis_functions

PairMTP.write = write_mtp                                  # potential.write(path)
PairMTP.fit = staticmethod(fit_mtp)

__all__ = ["PairMTP", "PairMTPExtrapolation", "MTP", "fit_mtp", "level_basis_functions"]
