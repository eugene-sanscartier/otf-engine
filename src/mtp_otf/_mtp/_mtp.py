"""Source wrapper for the compiled MTP extension.

The binary implementation lives in mtp._mtp_ext. This wrapper keeps the public
import path source-backed so Pylance can resolve it without suppressing
diagnostics.
"""

from importlib import import_module as _import_module

_ext = _import_module("._mtp_ext", __package__)

MTPPotential = _ext.MTPPotential

__all__ = ["MTPPotential"]
