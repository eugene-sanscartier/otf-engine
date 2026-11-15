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


@pytest.fixture(scope="session")
def recapture(request):
    return request.config.getoption("--recapture")


@pytest.fixture(scope="session")
def mlip3(request):
    return Path(request.config.getoption("--mlip3"))


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
