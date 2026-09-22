"""Source wrapper for the compiled MTP extension.

The binary implementation lives in otf_engine._mtp._mtp_ext. This wrapper keeps
the public import path source-backed so Pylance can resolve it without
suppressing diagnostics.

The three classes mirror the C++ hierarchy:
    PairMTP                 energy, forces, virial
    PairMTPExtrapolation    + information vectors and extrapolation grades
    MTPTraining             + derivatives w.r.t. the coefficients
"""

from importlib import import_module as _import_module

_ext = _import_module("._mtp_ext", __package__)

PairMTP = _ext.PairMTP
PairMTPExtrapolation = _ext.PairMTPExtrapolation
MTPTraining = _ext.MTPTraining
NeighList = _ext.NeighList

__all__ = ["PairMTP", "PairMTPExtrapolation", "MTPTraining", "NeighList"]
