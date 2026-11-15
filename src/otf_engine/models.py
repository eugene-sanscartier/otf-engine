"""The kinds of potential, MTP and ETN: what the selection and the training call for each."""

from __future__ import annotations

from importlib import import_module
from typing import Callable

from ._core import Pair
from ._etn import ETNTraining, PairETN, train_etn, write_etn
from ._mtp import MTPTraining, train_mtp, write_mtp


class Model:
    """A kind of potential: its classes, its trainers and its selection rules."""

    training: type                         # the training class, constructed from a potential file
    write: Callable                        # writes a potential to a file
    train_on_threads: Callable             # the trainer on threads
    mpi_module: str                        # the module of the trainer over MPI
    mpi_train: str                         # the trainer's name there
    threshold_swap: float | None = None    # MaxVol.threshold_swap of the model's searches
    retrain_pass: bool = True              # whether select_add searches the training set again after the candidates

    @classmethod
    def train_over_mpi(cls) -> Callable:
        """The trainer over MPI, imported here, so a process without a communicator never loads the MPI library."""
        trainer = getattr(import_module(cls.mpi_module), cls.mpi_train)
        return trainer


class MTP(Model):
    """mlip-3's Moment Tensor Potential, in .mtp and .almtp files."""

    training = MTPTraining
    write = staticmethod(write_mtp)
    train_on_threads = staticmethod(train_mtp)
    mpi_module = "otf_engine._mtp._mtp_mpi"
    mpi_train = "train_mtp"


class ETN(Model):
    """mlip-4's ETN, in .etn files, selected by the rules of mlip-4's CfgCalc."""

    training = ETNTraining
    write = staticmethod(write_etn)
    train_on_threads = staticmethod(train_etn)
    mpi_module = "otf_engine._etn._etn_mpi"
    mpi_train = "train_etn"
    threshold_swap = 1.00001    # mlip-4's MaximizeVol goes on swapping, once a search has swapped, while a grade exceeds this
    retrain_pass = False        # mlip-4's CfgCalc makes no training pass after the candidates


def model_of(potential: str | Pair) -> type[Model]:
    """The model of *potential*, a Pair or a potential file: an ETN's if the file's first line is ETN, else an MTP's."""
    if isinstance(potential, Pair):
        return ETN if isinstance(potential, PairETN) else MTP
    with open(potential, "rb") as f:    # the #MVS_v1.1 section is binary
        etn = f.readline().strip() == b"ETN"
    model = ETN if etn else MTP
    return model
