"""Read a training set file, .cfg or extxyz, and append structures to it."""

import ase.io.extxyz

from .io_cfg import read_cfg, write_cfg


def read_training_set(path) -> list:
    """Return the structures of the training set file *path*."""
    with open(path) as f:
        training_set = read_cfg(f) if str(path).endswith(".cfg") else list(ase.io.extxyz.read_extxyz(f, index=slice(None)))
    return training_set


def write_training_set(path, structures) -> None:
    """Append *structures* to the training set file *path*."""
    with open(path, "a") as f:
        if str(path).endswith(".cfg"):
            write_cfg(f, structures)
            return
        for atoms in structures:
            atoms.arrays.pop("id", None)
            atoms.arrays.pop("type", None)
        ase.io.extxyz.write_extxyz(f, structures)
