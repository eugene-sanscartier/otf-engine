"""Source wrapper for the compiled MTP extension.

The binary implementation lives in otf_engine._mtp._mtp_ext. This wrapper keeps
the public import path source-backed so Pylance can resolve it without
suppressing diagnostics.

The three classes mirror the C++ hierarchy:
    PairMTP                 energy, forces, virial
    PairMTPExtrapolation    + extrapolation grades           (a PairMTP)
    MTPTraining             + basis values and derivatives
                              w.r.t. the coefficients        (a PairMTP)

train_mtp fits an MTPTraining as mlip-3's `mlp train` does. Equations and
MaxVol select and grade against an active set as mlip-3's MaxVol does.
"""

from importlib import import_module as _import_module

_ext = _import_module("._mtp_ext", __package__)

PairMTP = _ext.PairMTP
PairMTPExtrapolation = _ext.PairMTPExtrapolation
MTPTraining = _ext.MTPTraining
NeighList = _ext.NeighList
train_mtp = _ext.train_mtp
Equations = _ext.Equations
MaxVol = _ext.MaxVol

__all__ = ["PairMTP", "PairMTPExtrapolation", "MTPTraining", "NeighList", "train_mtp", "Equations", "MaxVol"]
