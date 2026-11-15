"""Source wrapper for the compiled extension shared by the MTP and the ETN.

The binary implementation lives in otf_engine._core._core_ext, which _mtp_ext
and _etn_ext import for Pair and NeighList.

    NeighList          the neighbor list a Pair's compute takes
    Pair               the base of PairMTP and PairETN: coefficients, gradients, accessors
    TrainingStructure  a structure to fit, which the trainers and error_table take
    error_table   a potential's errors on structures, as train_mtp and train_etn log them
    Equations     the MaxVol equations of structures, from a training class's gradients
    MaxVol             an active set, graded against and searched
"""

from importlib import import_module as _import_module

_ext = _import_module("._core_ext", __package__)

NeighList = _ext.NeighList
Pair = _ext.Pair
TrainingStructure = _ext.TrainingStructure
error_table = _ext.error_table
Equations = _ext.Equations
MaxVol = _ext.MaxVol

__all__ = ["NeighList", "Pair", "TrainingStructure", "error_table", "Equations", "MaxVol"]
