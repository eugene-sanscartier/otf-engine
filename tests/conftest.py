"""Shared fixtures: the synthetic structures, the reference files, and the probes that --recapture reruns."""

import subprocess
from pathlib import Path

import numpy
import pytest

TESTS = Path(__file__).resolve().parent
DATA = TESTS / "data"
MTP_REFERENCE = TESTS / "mtp_reference"
ETN_REFERENCE = TESTS / "etn_reference"
CODE_LIBRARY = Path.home() / "Doctorat" / "code_library"


def pytest_addoption(parser):
    parser.addoption("--recapture", action="store_true", help="rerun mlp and mlip-4 through their probes and rewrite the stored outputs")
    parser.addoption("--mlip3", default=str(CODE_LIBRARY / "mlip-3"), help="mlip-3's source tree, with bin/mlp and lib/lib_mlip_cblas.a, for --recapture")
    parser.addoption("--mlip4-build", default=str(CODE_LIBRARY / "mlip-4" / "build-venv"), help="mlip-4's build directory, holding lib_mlip_4_interface.a, for --recapture")


def read_structures(name):
    from otf_engine.io_cfg import read_cfg

    with open(DATA / name) as f:
        structures = read_cfg(f)
    return structures


@pytest.fixture(scope="session")
def structures():
    """The 6 labelled structures of tests/data/structures.cfg, the first without stress."""
    return read_structures("structures.cfg")


@pytest.fixture(scope="session")
def candidates():
    """The 4 labelled structures of tests/data/candidates.cfg."""
    return read_structures("candidates.cfg")


def two_species(structures):
    """The structures with species 2 renamed 1, which data/two-species-08.mtp takes."""
    out = []
    for atoms in structures:
        copy = atoms.copy()
        copy.calc = atoms.calc
        copy.arrays["type_index"] = numpy.minimum(atoms.arrays["type_index"], 1)
        out += [copy]
    return out


def write_dump(path, frames, lammps_species=None, element=False):
    """Write *frames* as a LAMMPS text dump.

    lammps_species : the element of each LAMMPS type
    element        : whether the dump has an element column
    """
    from ase.calculators.lammps import Prism

    lines = []
    for step, atoms in enumerate(frames):
        prism = Prism(atoms.cell.array)
        xhi, yhi, zhi, xy, xz, yz = (float(x) for x in prism.get_lammps_prism())
        positions = prism.vector_to_lammps(atoms.positions, wrap=True).tolist()
        elements = atoms.get_chemical_symbols()
        types = atoms.arrays["type_index"] + 1 if lammps_species is None else [lammps_species.index(e) + 1 for e in elements]
        lines += ["ITEM: TIMESTEP", str(step), "ITEM: NUMBER OF ATOMS", str(len(atoms)), "ITEM: BOX BOUNDS xy xz yz pp pp pp"]
        lines += [f"{min(0.0, xy, xz, xy + xz)!r} {xhi + max(0.0, xy, xz, xy + xz)!r} {xy!r}", f"{min(0.0, yz)!r} {yhi + max(0.0, yz)!r} {xz!r}", f"0.0 {zhi!r} {yz!r}"]
        lines += ["ITEM: ATOMS id type element x y z" if element else "ITEM: ATOMS id type x y z"]
        lines += [f"{i + 1} {t} {e + ' ' if element else ''}{p[0]!r} {p[1]!r} {p[2]!r}" for i, (t, e, p) in enumerate(zip(types, elements, positions))]
    path.write_text("\n".join(lines) + "\n")


def several_chunks(atoms):
    """atoms repeated into a cell of at least 130 atoms, more than two chunks of central atoms of any build."""
    repeats = int(numpy.ceil((130 / len(atoms)) ** (1 / 3)))
    return atoms.repeat((repeats, repeats, repeats))


def central_atom_lists(nl):
    """One neighbor list per central atom of nl, over the same atoms."""
    from otf_engine._core import NeighList

    numneigh = numpy.asarray(nl.numneigh)
    starts = numpy.concatenate([[0], numpy.cumsum(numneigh)])
    firstneigh, displacements = numpy.asarray(nl.firstneigh), numpy.asarray(nl.displacements)
    for ii, i in enumerate(numpy.asarray(nl.ilist)):
        begin, end = starts[ii], starts[ii + 1]
        yield NeighList(nl.types, numpy.array([i], dtype=numpy.int32), numneigh[ii : ii + 1], firstneigh[begin:end], displacements[begin:end])


def compute_by_atom(potential, nl):
    """compute's energy, per-atom energies, forces and virial summed over one call per central atom of nl."""
    total = {"energy": 0.0, "eatom": 0.0, "forces": 0.0, "virial": 0.0}
    for single in central_atom_lists(nl):
        result = potential.compute(single)
        for key in total:
            total[key] = total[key] + numpy.asarray(result[key])
    return total


def assert_compute_by_atom(potential, nl):
    """compute over every central atom of nl at once, in chunks, gives the sum of compute over each atom alone."""
    whole = {key: numpy.asarray(value) for key, value in potential.compute(nl).items() if key in ("energy", "eatom", "forces", "virial")}
    by_atom = compute_by_atom(potential, nl)
    for key, value in whole.items():
        scale = numpy.max(numpy.abs(value))
        assert numpy.max(numpy.abs(by_atom[key] - value)) <= 1e-12 * scale, key


def assert_loss_grad_by_atom(potential, nl, seed=0):
    """eval_loss_grad over every central atom of nl at once, in chunks, gives the sum of eval_loss_grad over each atom
    alone, for seeded derivatives of the loss with respect to the energy, forces and virial."""
    rng = numpy.random.default_rng(seed)
    dloss_denergy, dloss_dforces, dloss_dvirial = float(rng.standard_normal()), rng.standard_normal((nl.n_atoms, 3)), rng.standard_normal(6)
    whole = numpy.asarray(potential.eval_loss_grad(nl, dloss_denergy, dloss_dforces, dloss_dvirial))
    by_atom = sum(numpy.asarray(potential.eval_loss_grad(single, dloss_denergy, dloss_dforces, dloss_dvirial)) for single in central_atom_lists(nl))
    assert numpy.max(numpy.abs(by_atom - whole)) <= 1e-12 * numpy.max(numpy.abs(whole))


@pytest.fixture(scope="session")
def recapture(request):
    return request.config.getoption("--recapture")


@pytest.fixture(scope="session")
def mlip3(request):
    return Path(request.config.getoption("--mlip3"))


@pytest.fixture(scope="session")
def mlip4_build(request):
    return Path(request.config.getoption("--mlip4-build"))


@pytest.fixture(scope="session")
def mlip3_probe(recapture, mlip3, tmp_path_factory):
    """mlip3_probe.cpp compiled with mlip-3's serial sources, under --recapture only."""
    if not recapture:
        return None
    probe = tmp_path_factory.mktemp("mlip3") / "mlip3_probe"
    sources = [str(p) for d in ("src", "src/common", "src/drivers") for p in sorted((mlip3 / d).glob("*.cpp"))]
    subprocess.run(["g++", "-O3", "-std=c++11", "-w", f"-I{mlip3 / 'src'}", f"-I{mlip3 / 'src' / 'common'}", str(MTP_REFERENCE / "mlip3_probe.cpp"), *sources, str(mlip3 / "lib" / "lib_mlip_cblas.a"), "-lgfortran", "-o", str(probe)], check=True)
    return probe


def parse_probe(printed):
    """The "@ NAME N : V..." lines of a probe's output, as (NAME, values) in order."""
    results = []
    for line in printed.splitlines():
        if line.startswith("@ "):
            head, values = line[2:].split(":", 1)
            name, count = head.split()
            results += [(name, numpy.array(values.split(), dtype=float).reshape(int(count)))]
    return results


class StoredOutputs:
    """A reference's outputs in an npz file, read by key; under --recapture the keys set replace theirs and the file is rewritten."""

    def __init__(self, path, recapture):
        self.path = path
        self.recapture = recapture
        self.arrays = dict(numpy.load(path)) if path.exists() else {}

    def __getitem__(self, key):
        return self.arrays[key]

    def __setitem__(self, key, value):
        self.arrays[key] = numpy.asarray(value)

    def keys_under(self, prefix):
        return sorted(k for k in self.arrays if k.startswith(prefix + "/"))

    def replace_under(self, prefix, outputs):
        """Sets the keys under prefix to outputs, (name, values) pairs in order, dropping the old ones."""
        for key in self.keys_under(prefix):
            del self.arrays[key]
        for i, (name, values) in enumerate(outputs):
            self[f"{prefix}/{i:03d}_{name}"] = values

    def save(self):
        if self.recapture:
            numpy.savez_compressed(self.path, **self.arrays)
