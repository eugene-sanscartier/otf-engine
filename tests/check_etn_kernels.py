"""Parity gate for the ETN C++ layer against mlip-4.

Every tensor `ETN` computes is compared with golden values in `etn_reference/`
that mlip-4 computed: the radial basis and its derivatives, the real spherical
harmonics and their gradients, the dense real and imaginary 3-j tables, and,
for a 3-core and a 5-core potential on two structures of the NiSi system, each
atom's input vector and bond vectors, the site energies, forces and virial.

    python tests/check_etn_kernels.py
        check this build against the golden values

    python tests/check_etn_kernels.py --capture --cfg tmp/set.cfg --mlip4-build DIR
        rewrite the golden values from mlip-4, whose build directory DIR holds
        lib_mlip_4_interface.a; the probe `etn_reference/mlip4_probe.cpp` is
        compiled against it, and the structures are the first two of the cfg.

The harmonics come from mlip-4's table in src/pots/etn/angular_basis.cpp; the
rest from the probe, which runs mlip-4's own ETN program.
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy
from ase import Atoms
from numpy import float64, int32

from otf_engine._etn import ETN, real_harmonics, real_wigner_3j, write_etn
from otf_engine._mtp.neighbors import neighbors

REFERENCE = Path(__file__).parent / "etn_reference"
GOLDEN = REFERENCE / "kernels.npz"
PROBE_SOURCE = REFERENCE / "mlip4_probe.cpp"
TOL = 1e-12

SPECIES_COUNT = 3
MIN_DIST, MAX_DIST, RADIAL_BASIS_SIZE = 1.0, 5.0, 6
SPECIES_RANKS, RADIAL_FUNCS_COUNTS, RADIAL_RANKS = [2, 2, 1], [6, 4, 3], [3, 2, 2]
POTENTIALS = {
    "etn-3": [[1], [3, 2, 1], [3, 2, 1], [1]],
    "etn-5": [[1], [2, 2, 1], [2, 2, 1], [2, 2, 1], [2, 2, 1], [1]],
}
PROBE_DISTANCES = [1.0, 1.5, 2.5, 4.0, 4.9]
WIGNER_L_COUNT = 5
HARMONICS_L_COUNT = 5
DIRECTIONS = numpy.array([[0.3, -0.5, 0.7], [0.0, 0.0, 1.0], [-1.0, 2.0, 0.5], [0.6, 0.8, 0.0]])


def directions():
    return DIRECTIONS / numpy.linalg.norm(DIRECTIONS, axis=1)[:, None]


def build_potentials():
    """The two test potentials, with seeded coefficients."""
    built = {}
    for k, (name, bond_ranks) in enumerate(POTENTIALS.items()):
        pot = ETN(SPECIES_COUNT, SPECIES_RANKS, RADIAL_FUNCS_COUNTS, RADIAL_RANKS, bond_ranks, RADIAL_BASIS_SIZE, MIN_DIST, MAX_DIST)
        pot.set_coeffs(numpy.random.default_rng(k).standard_normal(pot.get_coeff_count()) * 0.3)
        built[name] = pot
    return built


def structures(golden):
    """The structures stored in the golden values."""
    built, s = [], 0
    while f"structure/{s}/positions" in golden:
        atoms = Atoms(numbers=numpy.full(len(golden[f"structure/{s}/types"]), 1), positions=golden[f"structure/{s}/positions"], cell=golden[f"structure/{s}/cell"], pbc=True)
        built += [(atoms, numpy.asarray(golden[f"structure/{s}/types"], dtype=int32))]
        s += 1
    return built


def outputs(potentials, cells):
    """Every tensor of this build, keyed as the golden values."""
    out = {}
    pot = next(iter(potentials.values()))

    rb_vals, rb_ders = zip(*(pot.eval_radial_basis(d) for d in PROBE_DISTANCES))
    out["rb_vals"], out["rb_ders"] = numpy.array(rb_vals), numpy.array(rb_ders)

    for k, u in enumerate(directions()):
        vals, grads = real_harmonics(HARMONICS_L_COUNT, u)
        out[f"ylm/{k}/vals"], out[f"ylm/{k}/grads"] = numpy.asarray(vals), numpy.asarray(grads)

    lm = WIGNER_L_COUNT * WIGNER_L_COUNT
    w_re, w_im = numpy.zeros((lm, lm, lm)), numpy.zeros((lm, lm, lm))
    for l1 in range(WIGNER_L_COUNT):
        for l2 in range(WIGNER_L_COUNT):
            for l3 in range(WIGNER_L_COUNT):
                table = w_re if (l1 + l2 + l3) % 2 == 0 else w_im
                for m1 in range(-l1, l1 + 1):
                    for m2 in range(-l2, l2 + 1):
                        for m3 in range(-l3, l3 + 1):
                            table[l1 * l1 + l1 + m1, l2 * l2 + l2 + m2, l3 * l3 + l3 + m3] = real_wigner_3j(l1, m1, l2, m2, l3, m3)
    out["wigner_re"], out["wigner_im"] = w_re, w_im

    for name, pot in potentials.items():
        for s, (atoms, types) in enumerate(cells):
            result = pot.compute(neighbors(atoms, MAX_DIST, types=types))
            prefix = f"{name}/s{s}/"
            for key in ("energy", "eatom", "forces", "virials", "inputs"):
                out[prefix + key] = numpy.asarray(result[key], dtype=float64)
            for b, (re_part, im_part) in enumerate(zip(result["bonds"], result["bonds_im"])):
                out[prefix + f"bonds/{b}"] = numpy.asarray(re_part)
                out[prefix + f"bonds_im/{b}"] = numpy.asarray(im_part)
    return out


# ---------------------------------------------------------------- capture


def table_harmonics(source):
    """mlip-4's Y_lm and their gradients tangent to the sphere, from its monomial table."""
    text = Path(source).read_text()
    moments = numpy.array([int(x) for x in re.findall(r"-?\d+", re.search(r"MOMENTS = \{(.*?)\};", text, re.S).group(1))]).reshape(-1, 4)
    coeffs = numpy.array([float(x) for x in re.search(r"COEFFS\s*=\s*\{(.*?)\};", text, re.S).group(1).replace("{", "").replace("}", "").split(",") if x.strip()])
    offsets = [int(x) for x in re.search(r"OFFSETS = \{(.*?)\};", text).group(1).split(",")]

    out = {}
    for k, u in enumerate(directions()):
        vals, grads = numpy.zeros(HARMONICS_L_COUNT ** 2), numpy.zeros((HARMONICS_L_COUNT ** 2, 3))
        for (index, a, b, c), w in zip(moments[: offsets[HARMONICS_L_COUNT - 1]], coeffs):
            powers = numpy.array([a, b, c])
            vals[index] += w * numpy.prod(u ** powers)
            for axis in range(3):
                if powers[axis] > 0:
                    lowered = powers.copy()
                    lowered[axis] -= 1
                    grads[index, axis] += w * powers[axis] * numpy.prod(u ** lowered)
        grads -= (grads @ u)[:, None] * u
        out[f"ylm/{k}/vals"], out[f"ylm/{k}/grads"] = vals, grads
    return out


def compile_probe(build_dir, workdir):
    """Compile the probe against mlip-4's build, with the include flags of its ETN module."""
    flags = (Path(build_dir) / "src/python/CMakeFiles/mlip_etn.dir/flags.make").read_text()
    includes = re.search(r"^CXX_INCLUDES = (.*)$", flags, re.M).group(1).split()
    probe = Path(workdir) / "mlip4_probe"
    subprocess.run(["g++", "-O2", "-std=gnu++17", *includes, str(PROBE_SOURCE), str(Path(build_dir) / "lib_mlip_4_interface.a"), "-lpthread", "-ldl", "-o", str(probe)], check=True)
    return probe


def run_probe(probe, pot, cells, workdir):
    """mlip-4's tensors for pot on cells, parsed from the probe's output."""
    lines = [f"{SPECIES_COUNT} {len(SPECIES_RANKS)} {pot.get_core_count()} {RADIAL_BASIS_SIZE} {MIN_DIST!r} {MAX_DIST!r}"]
    lines += [" ".join(map(str, SPECIES_RANKS)), " ".join(map(str, RADIAL_FUNCS_COUNTS)), " ".join(map(str, RADIAL_RANKS))]
    lines += [f"{len(ranks)} " + " ".join(map(str, ranks)) for ranks in pot.get_bond_ranks()]
    coeffs = pot.get_coeffs()
    lines += [str(len(coeffs)), " ".join(repr(float(x)) for x in coeffs)]
    lines += [str(len(PROBE_DISTANCES)), " ".join(map(repr, PROBE_DISTANCES)), str(WIGNER_L_COUNT), str(len(cells))]
    for atoms, types in cells:
        lines += [str(len(atoms))]
        lines += [" ".join(repr(float(x)) for x in row) for row in atoms.cell.array]
        lines += [f"{t} " + " ".join(repr(float(x)) for x in p) for t, p in zip(types, atoms.positions)]
    probe_file = Path(workdir) / "probe.txt"
    probe_file.write_text("\n".join(lines) + "\n")

    printed = subprocess.run([str(probe), str(probe_file)], check=True, capture_output=True, text=True).stdout
    tensors = {}
    for line in printed.splitlines():
        head, values = line.split(" :")
        name, *shape = head.split()
        tensors[name] = numpy.array([float(x) for x in values.split()]).reshape([int(n) for n in shape])
    return tensors


def capture(args):
    from otf_engine.io_cfg import read_cfg

    with open(args.cfg) as f:
        configs = read_cfg(f)[:2]
    cells = [(atoms, numpy.asarray(atoms.arrays["type_index"], dtype=int32)) for atoms in configs]

    source_root = re.search(r"^CMAKE_HOME_DIRECTORY:INTERNAL=(.*)$", (Path(args.mlip4_build) / "CMakeCache.txt").read_text(), re.M).group(1)
    golden = table_harmonics(Path(source_root) / "src/pots/etn/angular_basis.cpp")
    for s, (atoms, types) in enumerate(cells):
        golden[f"structure/{s}/positions"] = atoms.positions
        golden[f"structure/{s}/cell"] = atoms.cell.array
        golden[f"structure/{s}/types"] = types

    potentials = build_potentials()
    with tempfile.TemporaryDirectory() as workdir:
        probe = compile_probe(args.mlip4_build, workdir)
        for name, pot in potentials.items():
            write_etn(pot, REFERENCE / f"{name}.etn")
            tensors = run_probe(probe, pot, cells, workdir)
            for key in ("rb_vals", "rb_ders", "wigner_re", "wigner_im"):
                golden[key] = tensors[key]
            for s in range(len(cells)):
                for key, value in tensors.items():
                    if key.startswith(f"s{s}/"):
                        golden[f"{name}/{key}"] = value

    numpy.savez_compressed(GOLDEN, **golden)
    print(f"captured {len(golden)} arrays to {GOLDEN} ({GOLDEN.stat().st_size / 1024:.0f} KB)")


# ---------------------------------------------------------------- check


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture", action="store_true", help="rewrite the golden values from mlip-4")
    parser.add_argument("--cfg", help="cfg whose first two structures the capture uses")
    parser.add_argument("--mlip4-build", help="mlip-4 build directory, for the capture")
    parser.add_argument("--tol", type=float, default=TOL, help=f"relative tolerance (default {TOL:g})")
    args = parser.parse_args()

    if args.capture:
        if not args.cfg or not args.mlip4_build:
            parser.error("--capture needs --cfg and --mlip4-build")
        capture(args)
        return 0

    golden = numpy.load(GOLDEN)
    potentials = {name: ETN(str(REFERENCE / f"{name}.etn")) for name in POTENTIALS}
    collected = outputs(potentials, structures(golden))
    failures, worst = [], 0.0

    for name in sorted(collected):
        if name not in golden.files:
            failures += [f"{name}: absent from the golden values"]
            continue
        want, got = golden[name], collected[name]
        if got.shape != want.shape:
            failures += [f"{name}: shape {got.shape} != {want.shape}"]
            continue
        rel = float(numpy.abs(got - want).max()) / max(float(numpy.abs(want).max()), 1e-300)
        worst = max(worst, rel)
        if rel > args.tol:
            failures += [f"{name}: rel {rel:.3e}"]

    for name in sorted(set(golden.files) - set(collected)):
        if not name.startswith("structure/"):
            failures += [f"{name}: missing from this build"]

    print(f"{len(collected)} arrays checked, worst relative difference {worst:.3e}, tolerance {args.tol:g}")
    for line in failures[:20]:
        print(f"  FAIL {line}")
    if len(failures) > 20:
        print(f"  ... and {len(failures) - 20} more")
    print("every tensor matches mlip-4" if not failures else f"{len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
