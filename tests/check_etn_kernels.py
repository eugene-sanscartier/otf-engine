"""Parity gate for the ETN C++ layer against mlip-4.

Every tensor `PairETN` computes is compared with golden values in `etn_reference/`
that mlip-4 computed: the radial basis and its derivatives, the real spherical
harmonics and their gradients, the dense real and imaginary 3-j tables, and,
for six potentials on three NiSi structures, the energy, forces and virial,
each atom's site energy, the first atoms' input and bond vectors, the gradient
of a loss with seeded derivatives, and the first atoms' site-energy gradients. The
potentials span 1 to 5 cores, l up to 4, bonds with more l than the input,
zero ranks and a single species. Each array is compared relative to its
largest entry, and each row, at a looser tolerance, relative to its own.

Then, without mlip-4: forces, virial and the loss gradient against finite
differences, invariance under rotation and reflection, the .etn round trip,
and the untrained file.

    python tests/check_etn_kernels.py
        check this build

    python tests/check_etn_kernels.py --capture --cfg tmp/set.cfg --mlip4-build DIR
        rewrite the golden values from mlip-4, whose build directory DIR holds
        lib_mlip_4_interface.a; the probe `etn_reference/mlip4_probe.cpp` is
        compiled against it and runs mlip-4's own ETN program.
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

from otf_engine._etn import ETNTraining, PairETN, real_harmonics, real_wigner_3j, write_etn
from otf_engine._mtp.neighbors import neighbors
from otf_engine._mtp.selftest import check_forces, check_invariance, check_virial

REFERENCE = Path(__file__).parent / "etn_reference"
GOLDEN = REFERENCE / "kernels.npz"
PROBE_SOURCE = REFERENCE / "mlip4_probe.cpp"
TOL = 1e-12
ROW_TOL = 1e-9  # rows whose terms cancel, such as small net forces, round to 1e-11 of themselves
SELFTEST_TOL = 1e-6

# Structures of the cfg: one Ni; two Ni 3.7 A apart; a distant frame
STRUCTURES = [0, 2410, 17763]
TENSOR_ATOMS = 24  # input and bond vectors are kept for the first atoms, the Ni among them
GRAD_ATOMS = 6  # site-energy gradients are kept for the first atoms
MIN_DIST, MAX_DIST, RADIAL_BASIS_SIZE = 1.0, 5.0, 6
# name: species count, species ranks, radial function counts, radial ranks (per l of the input), bond ranks
POTENTIALS = {
    "etn-1": (3, [2, 1], [6, 4], [3, 2], [[1], [1]]),
    "etn-2": (3, [2, 1], [6, 4], [3, 2], [[1], [2, 2, 1, 1], [1]]),
    "etn-3": (3, [2, 2, 1], [6, 4, 3], [3, 2, 2], [[1], [3, 2, 1], [3, 2, 1], [1]]),
    "etn-4": (3, [2, 0, 1, 1, 1], [6, 4, 3, 2, 2], [3, 2, 0, 1, 1], [[1], [2, 0, 1, 1, 1], [2, 1, 1, 1, 1], [2, 0, 1, 1, 1], [1]]),
    "etn-5": (3, [2, 2, 1], [6, 4, 3], [3, 2, 2], [[1], [2, 2, 1], [2, 2, 1], [2, 2, 1], [2, 2, 1], [1]]),
    "etn-3-single": (1, [2, 1, 1], [6, 4, 3], [3, 2, 1], [[1], [2, 2, 1], [2, 2, 1], [1]]),
}
PROBE_DISTANCES = [1.0, 1.5, 2.5, 4.0, 4.9]
WIGNER_L_COUNT = 5
HARMONICS_L_COUNT = 5
DIRECTIONS = numpy.array([[0.3, -0.5, 0.7], [0.0, 0.0, 1.0], [-1.0, 2.0, 0.5], [0.6, 0.8, 0.0]])


def directions():
    return DIRECTIONS / numpy.linalg.norm(DIRECTIONS, axis=1)[:, None]


def coeff_blocks(pot):
    """The coefficient blocks of pot as (kind, argument), in mlip-4's params order."""
    l_count, core_count = len(pot.get_species_ranks()), pot.get_core_count()
    blocks = [("radial", l) for l in range(l_count)] + [("species_pair", l) for l in range(l_count)]
    blocks += [("core", (i, False)) for i in range(core_count)]
    blocks += [("core", (i, True)) for i in range(1, core_count - 1) if core_count > 3]
    return blocks + [("species", None)]


def get_block(pot, kind, arg):
    return {"radial": lambda: pot.get_radial_coeffs(arg), "species_pair": lambda: pot.get_species_pair_coeffs(arg),
            "core": lambda: pot.get_core_coeffs(*arg), "species": pot.get_species_coeffs}[kind]()


def params(pot):
    """The coefficients of pot as mlip-4's params vector."""
    return numpy.concatenate([get_block(pot, kind, arg) for kind, arg in coeff_blocks(pot)])


def set_params(pot, values):
    """Split mlip-4's params vector into pot's coefficient blocks."""
    offset = 0
    for kind, arg in coeff_blocks(pot):
        size = len(get_block(pot, kind, arg))
        block = values[offset : offset + size]
        offset += size
        if kind == "radial":
            pot.set_radial_coeffs(arg, block)
        elif kind == "species_pair":
            pot.set_species_pair_coeffs(arg, block)
        elif kind == "core":
            pot.set_core_coeffs(arg[0], block, imaginary=arg[1])
        else:
            pot.set_species_coeffs(block)


def template(name):
    """The untrained .etn file of a test potential."""
    species_count, species_ranks, radial_funcs_counts, radial_ranks, bond_ranks = POTENTIALS[name]
    braces = lambda values: "{" + ", ".join(map(str, values)) + "}"
    lines = ["ETN", "version = 1.0.0", f"potential_name = {name}", f"species_count = {species_count}", "radial_basis_type = RBCinf",
             f"\tmin_dist = {MIN_DIST}", f"\tmax_dist = {MAX_DIST}", f"\tradial_basis_size = {RADIAL_BASIS_SIZE}",
             f"species_ranks = {braces(species_ranks)}", f"radial_funcs_counts = {braces(radial_funcs_counts)}",
             f"radial_ranks = {braces(radial_ranks)}", f"core_count = {len(bond_ranks) - 1}", "bond_ranks"]
    return "\n".join(lines + ["\t" + braces(ranks) for ranks in bond_ranks]) + "\n"


def build_potentials(workdir):
    """The test potentials, from their templates with seeded coefficients."""
    built = {}
    for k, name in enumerate(POTENTIALS):
        path = Path(workdir) / f"{name}.etn"
        path.write_text(template(name))
        pot = PairETN(str(path))
        set_params(pot, numpy.random.default_rng(k).standard_normal(pot.get_coeff_count()) * 0.3)
        pot.set_trained(True)
        built[name] = pot
    return built


def structures(golden):
    """The structures stored in the golden values, with their species."""
    built, s = [], 0
    while f"structure/{s}/positions" in golden:
        types = numpy.asarray(golden[f"structure/{s}/types"], dtype=int32)
        atoms = Atoms(numbers=numpy.full(len(types), 1), positions=golden[f"structure/{s}/positions"], cell=golden[f"structure/{s}/cell"], pbc=True)
        built += [(atoms, types)]
        s += 1
    return built


def species_of(pot, types):
    """The species the potential sees: every atom is species 0 for a single-species potential."""
    return types if pot.get_species_count() > 1 else numpy.zeros_like(types)


def loss_seeds(s, n_atoms):
    """Seeded derivatives of a loss with respect to structure s's energy, virial (xx, yy, zz, xy, xz, yz) and forces."""
    rng = numpy.random.default_rng(100 + s)
    return float(rng.standard_normal()), rng.standard_normal(6), rng.standard_normal((n_atoms, 3))


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
            nl = neighbors(atoms, MAX_DIST, types=species_of(pot, types))
            result = pot.compute(nl)
            prefix = f"{name}/s{s}/"
            for key in ("energy", "eatom", "forces", "virials"):
                out[prefix + key] = numpy.asarray(result[key], dtype=float64)
            out[prefix + "inputs"] = numpy.asarray(result["inputs"])[:TENSOR_ATOMS]
            for b, (re_part, im_part) in enumerate(zip(result["bonds"], result["bonds_im"])):
                out[prefix + f"bonds/{b}"] = numpy.asarray(re_part)[:TENSOR_ATOMS]
                out[prefix + f"bonds_im/{b}"] = numpy.asarray(im_part)[:TENSOR_ATOMS]
            dloss_denergy, dloss_dvirial, dloss_dforces = loss_seeds(s, len(atoms))
            out[prefix + "loss_grad"] = numpy.asarray(pot.eval_loss_grad(nl, dloss_denergy, dloss_dforces, dloss_dvirial))
            out[prefix + "grad"] = numpy.asarray(pot.eval_grad(nl))[:GRAD_ATOMS]
    return out


def check_loss_grad(pot, atoms, types, step=1e-5):
    """Largest relative error of eval_loss_grad along a random direction in each coefficient block, against central differences."""
    nl = neighbors(atoms, MAX_DIST, types=species_of(pot, types))
    dloss_denergy, dloss_dvirial, dloss_dforces = loss_seeds(0, len(atoms))
    coeffs = pot.get_coeffs()

    def loss(c):
        pot.set_coeffs(c)
        result = pot.compute(nl)
        return dloss_denergy * result["energy"] + (dloss_dforces * result["forces"]).sum() + (dloss_dvirial * result["virials"]).sum()

    pot.set_coeffs(coeffs)
    grad = pot.eval_loss_grad(nl, dloss_denergy, dloss_dforces, dloss_dvirial)
    rng = numpy.random.default_rng(0)
    sizes = [len(get_block(pot, kind, arg)) for kind, arg in coeff_blocks(pot)]
    worst, offset = 0.0, 0
    for size in sizes:
        direction = numpy.zeros_like(coeffs)
        direction[offset : offset + size] = rng.standard_normal(size)
        offset += size
        analytic = grad @ direction
        if size == 0 or analytic == 0.0:
            continue
        numeric = (loss(coeffs + step * direction) - loss(coeffs - step * direction)) / (2 * step)
        worst = max(worst, abs(numeric - analytic) / abs(analytic))
    pot.set_coeffs(coeffs)
    return worst


def relative_differences(got, want):
    """Largest difference relative to the array's largest entry, and relative to each row's, floored at 1e-6 of the array's."""
    got, want = numpy.atleast_2d(got), numpy.atleast_2d(want)
    if got.ndim > 2:
        got, want = got.reshape(len(got), -1), want.reshape(len(want), -1)
    array_max = max(float(numpy.abs(want).max(initial=0.0)), 1e-300)
    diff = numpy.abs(got - want)
    scale = numpy.maximum(numpy.abs(want).max(axis=1, initial=0.0), 1e-6 * array_max)
    return float(diff.max(initial=0.0)) / array_max, float((diff.max(axis=1, initial=0.0) / scale).max(initial=0.0))


def self_tests(potentials, cells):
    """Checks that need no mlip-4, as {name: error}."""
    atoms, types = cells[1]
    atoms = atoms.copy()
    results = {}
    for name in ("etn-4", "etn-5"):
        pot = potentials[name]
        atoms.arrays["type_index"] = species_of(pot, types)
        results[f"{name} forces"] = check_forces(pot, atoms)
        results[f"{name} virial"] = check_virial(pot, atoms)
        results[f"{name} invariance"] = check_invariance(pot, atoms)
    # The imaginary cores' derivatives are 1e-7 of the loss, below what differences resolve
    results["etn-3 loss gradient"] = check_loss_grad(potentials["etn-3"], *cells[1])

    with tempfile.TemporaryDirectory() as workdir:
        for name, pot in potentials.items():
            path = Path(workdir) / f"{name}.etn"
            write_etn(pot, path)
            back = PairETN(str(path))
            results[f"{name} round trip"] = float(numpy.abs(params(back) - params(pot)).max()) + (0.0 if back.is_trained() else 1.0)

        path = Path(workdir) / "untrained.etn"
        path.write_text(template("etn-5"))
        untrained = PairETN(str(path))
        initial = numpy.sin(10 + numpy.arange(untrained.get_coeff_count()) ** 2)
        write_etn(untrained, Path(workdir) / "untrained-back.etn")
        back = PairETN(str(Path(workdir) / "untrained-back.etn"))
        results["untrained initial values"] = float(numpy.abs(params(untrained) - initial).max()) + (1.0 if untrained.is_trained() else 0.0)
        results["untrained round trip"] = float(numpy.abs(params(back) - initial).max()) + (1.0 if back.is_trained() else 0.0)
    return results


# ---------------------------------------------------------------- capture


def compile_probe(build_dir, workdir):
    """Compile the probe against mlip-4's build, with the include flags of its ETN module."""
    flags = (Path(build_dir) / "src/python/CMakeFiles/mlip_etn.dir/flags.make").read_text()
    includes = re.search(r"^CXX_INCLUDES = (.*)$", flags, re.M).group(1).split()
    probe = Path(workdir) / "mlip4_probe"
    subprocess.run(["g++", "-O2", "-std=gnu++17", *includes, str(PROBE_SOURCE), str(Path(build_dir) / "lib_mlip_4_interface.a"), "-lpthread", "-ldl", "-o", str(probe)], check=True)
    return probe


def run_probe(probe, pot, cells, workdir):
    """mlip-4's tensors for pot on cells, parsed from the probe's output."""
    ints = lambda values: " ".join(map(str, values))
    floats = lambda values: " ".join(repr(float(x)) for x in values)
    lines = [f"{pot.get_species_count()} {len(pot.get_species_ranks())} {pot.get_core_count()} {RADIAL_BASIS_SIZE} {MIN_DIST!r} {MAX_DIST!r}"]
    lines += [ints(pot.get_species_ranks()), ints(pot.get_radial_funcs_counts()), ints(pot.get_radial_ranks())]
    lines += [f"{len(ranks)} {ints(ranks)}" for ranks in pot.get_bond_ranks()]
    coeffs = params(pot)
    lines += [str(len(coeffs)), floats(coeffs)]
    lines += [str(len(PROBE_DISTANCES)), floats(PROBE_DISTANCES), str(WIGNER_L_COUNT)]
    lines += [f"{HARMONICS_L_COUNT} {len(DIRECTIONS)}"] + [floats(u) for u in directions()]
    lines += [f"{TENSOR_ATOMS} {GRAD_ATOMS} {len(cells)}"]
    for s, (atoms, types) in enumerate(cells):
        lines += [str(len(atoms))] + [floats(row) for row in atoms.cell.array]
        lines += [f"{t} {floats(p)}" for t, p in zip(species_of(pot, types), atoms.positions)]
        dloss_denergy, dloss_dvirial, dloss_dforces = loss_seeds(s, len(atoms))
        lines += [floats([dloss_denergy]), floats(dloss_dvirial)] + [floats(row) for row in dloss_dforces]
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
        configs = read_cfg(f)
    cells = [(configs[k], numpy.asarray(configs[k].arrays["type_index"], dtype=int32)) for k in STRUCTURES]

    golden = {}
    for s, (atoms, types) in enumerate(cells):
        golden[f"structure/{s}/positions"] = atoms.positions
        golden[f"structure/{s}/cell"] = atoms.cell.array
        golden[f"structure/{s}/types"] = types

    with tempfile.TemporaryDirectory() as workdir:
        potentials = build_potentials(workdir)
        probe = compile_probe(args.mlip4_build, workdir)
        for name, pot in potentials.items():
            write_etn(pot, REFERENCE / f"{name}.etn")
            tensors = run_probe(probe, pot, cells, workdir)
            for key, value in tensors.items():
                if key.startswith("s"):
                    golden[f"{name}/{key}"] = value
                else:
                    golden[key] = value

    numpy.savez_compressed(GOLDEN, **golden)
    print(f"captured {len(golden)} arrays to {GOLDEN} ({GOLDEN.stat().st_size / 1024:.0f} KB)")


# ---------------------------------------------------------------- check


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture", action="store_true", help="rewrite the golden values from mlip-4")
    parser.add_argument("--cfg", help="cfg holding the structures, for the capture")
    parser.add_argument("--mlip4-build", help="mlip-4 build directory, for the capture")
    parser.add_argument("--tol", type=float, default=TOL, help=f"relative tolerance against mlip-4 (default {TOL:g})")
    args = parser.parse_args()

    if args.capture:
        if not args.cfg or not args.mlip4_build:
            parser.error("--capture needs --cfg and --mlip4-build")
        capture(args)
        return 0

    golden = numpy.load(GOLDEN)
    potentials = {name: ETNTraining(str(REFERENCE / f"{name}.etn")) for name in POTENTIALS}
    cells = structures(golden)
    collected = outputs(potentials, cells)
    failures, worst, worst_row = [], 0.0, 0.0

    for name in sorted(collected):
        if name not in golden.files:
            failures += [f"{name}: absent from the golden values"]
            continue
        want, got = golden[name], collected[name]
        if got.shape != want.shape:
            failures += [f"{name}: shape {got.shape} != {want.shape}"]
            continue
        rel, rel_row = relative_differences(got, want)
        worst, worst_row = max(worst, rel), max(worst_row, rel_row)
        if rel > args.tol or rel_row > ROW_TOL:
            failures += [f"{name}: rel {rel:.3e}, by row {rel_row:.3e}"]

    for name in sorted(set(golden.files) - set(collected)):
        if not name.startswith("structure/"):
            failures += [f"{name}: missing from this build"]

    print(f"{len(collected)} arrays checked against mlip-4, worst relative difference {worst:.3e} (tolerance {args.tol:g}), by row {worst_row:.3e} (tolerance {ROW_TOL:g})")

    for name, err in self_tests(potentials, cells).items():
        print(f"  {name:28s} {err:.3e}")
        if not err <= SELFTEST_TOL:
            failures += [f"{name}: {err:.3e} above {SELFTEST_TOL:g}"]

    for line in failures[:20]:
        print(f"  FAIL {line}")
    if len(failures) > 20:
        print(f"  ... and {len(failures) - 20} more")
    print("every check passes" if not failures else f"{len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
