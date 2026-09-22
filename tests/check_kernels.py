"""Kernel-equivalence gate for the MTP C++ layer.

Every entry point of `PairMTP`, `PairMTPExtrapolation` and `MTPTraining` is run
against golden values in `kernel_reference/`, on a level-16 3-species potential.
The forward pass exists in four hand-written copies, one per `compute`-shaped
method; this is what catches a transcription slip between them.

    python tests/check_kernels.py             check against the golden values
    python tests/check_kernels.py --capture   rewrite them from this build

Capture only from a build that has already passed, and say in the commit why the
values moved.
"""

import argparse
import sys
from pathlib import Path

import numpy
from ase import Atoms
from numpy import float64, int32

from otf_engine._mtp import MTPTraining
from otf_engine._mtp.neighbors import neighbors

REFERENCE = Path(__file__).parent / "kernel_reference"
POTENTIAL = REFERENCE / "potential-16.mtp"
GOLDEN = REFERENCE / "kernels.npz"
TOL = 1e-12

BASIS_TYPES = ["RBChebyshev", "RBChebyshev_repuls", "BChebyshev", "BChebyshev_repuls", "RBTaylor"]
PROBE_DISTANCES = [1.0, 1.5, 2.5, 4.0, 4.99]


def structures(min_dist, max_dist):
    """The cells the kernels are evaluated on.

    Built here rather than stored so the golden file holds only outputs.  Three
    cells, each small enough to keep that file small and periodic enough that a
    `max_dist` cutoff wraps: mixed species, single species, and one holding a
    pair at `min_dist` to exercise the short-range end of the radial basis.
    """
    rng = numpy.random.default_rng(0)
    built = []

    for n_species in (3, 1):
        cell = 9.0
        positions, types = [], []
        while len(positions) < 12:
            trial = rng.uniform(0, cell, 3)
            deltas = numpy.asarray(positions) - trial if positions else numpy.zeros((0, 3))
            deltas -= cell * numpy.round(deltas / cell)
            if len(positions) == 0 or numpy.linalg.norm(deltas, axis=1).min() > min_dist * 1.05:
                positions += [trial]
                types += [len(positions) % n_species]
        built += [(numpy.asarray(positions), numpy.asarray(types, dtype=int32), cell)]

    # A pair exactly at min_dist, where the clamped bases diverge from the plain ones.
    positions, types, cell = built[0][0].copy(), built[0][1].copy(), built[0][2]
    positions[1] = positions[0] + numpy.array([min_dist, 0.0, 0.0])
    built += [(positions, types, cell)]

    return [(Atoms(numbers=numpy.full(len(p), 26), positions=p, cell=[c, c, c], pbc=True), t) for p, t, c in built]


def outputs(pot, cutoff):
    """Every kernel's output on every structure, keyed for the golden file."""
    out = {}

    out["scalars"] = numpy.array([pot.get_coeff_count(), pot.get_species_count(), pot.get_radial_func_count(),
                                  pot.get_radial_basis_size(), pot.get_alpha_scalar_count(), pot.get_alpha_moment_count(),
                                  pot.get_alpha_index_basic_count(), pot.get_alpha_index_times_count(),
                                  pot.get_radial_coeff_count()], dtype=float64)
    out["cutoffs"] = numpy.array([pot.get_min_cutoff(), pot.get_max_cutoff(), pot.get_scaling()], dtype=float64)
    out["radial_basis_coeffs"] = numpy.asarray(pot.get_radial_basis_coeffs(), dtype=float64)
    out["alpha_index_basic"] = numpy.asarray(pot.get_alpha_index_basic(), dtype=float64)
    out["alpha_index_times"] = numpy.asarray(pot.get_alpha_index_times(), dtype=float64)
    out["alpha_moment_mapping"] = numpy.asarray(pot.get_alpha_moment_mapping(), dtype=float64)
    out["linear_coeffs"] = numpy.asarray(pot.get_linear_coeffs(), dtype=float64)
    out["species_coeffs"] = numpy.asarray(pot.get_species_coeffs(), dtype=float64)

    for dist in PROBE_DISTANCES:
        vals, ders = pot.eval_radial_basis(dist)
        out[f"rb/{dist}/vals"] = numpy.asarray(vals, dtype=float64)
        out[f"rb/{dist}/ders"] = numpy.asarray(ders, dtype=float64)

    for i, (atoms, types) in enumerate(structures(pot.get_min_cutoff(), cutoff)):
        nl = neighbors(atoms, cutoff, types=types)

        result = pot.compute(nl, compute_virials=True, compute_eatom=True)
        out[f"{i}/energy"] = numpy.asarray(result["energy"], dtype=float64)
        out[f"{i}/forces"] = numpy.asarray(result["forces"], dtype=float64)
        out[f"{i}/virials"] = numpy.asarray(result["virials"], dtype=float64)
        out[f"{i}/eatom"] = numpy.asarray(result["eatom"], dtype=float64)

        out[f"{i}/basis"] = numpy.asarray(pot.eval_basis(nl), dtype=float64)
        out[f"{i}/grad"] = numpy.asarray(pot.eval_grad(nl), dtype=float64)

        for name, kernel in (("all", pot.eval_grad_all), ("rad", pot.eval_grad_radial), ("lin", pot.eval_grad_linear)):
            eg, fg, vg = kernel(nl, True)
            out[f"{i}/{name}_eg"] = numpy.asarray(eg, dtype=float64)
            out[f"{i}/{name}_fg"] = numpy.asarray(fg, dtype=float64)
            out[f"{i}/{name}_vg"] = numpy.asarray(vg, dtype=float64)

        # grade() reaches the grading branch of PairMTPExtrapolation::compute, which
        # eval_grad leaves unrun. The active set is synthetic and seeded — the gate
        # checks that the kernel reproduces, not that the grades mean anything.
        n_coeffs = pot.get_coeff_count()
        invA = numpy.random.default_rng(1).standard_normal((n_coeffs, n_coeffs))
        for mode, label in ((False, "nbh"), (True, "cfg")):
            pot.set_active_set(invA, mode)
            grades, cfg_grade = pot.grade(nl)
            out[f"{i}/grade_{label}"] = numpy.asarray(grades, dtype=float64)
            out[f"{i}/grade_{label}_cfg"] = numpy.asarray(cfg_grade, dtype=float64)

        energy, forces, virials, eg, fg, vg = pot.compute_with_radial_grad(nl, True)
        out[f"{i}/cwrg_energy"] = numpy.asarray(energy, dtype=float64)
        out[f"{i}/cwrg_forces"] = numpy.asarray(forces, dtype=float64)
        out[f"{i}/cwrg_virials"] = numpy.asarray(virials, dtype=float64)
        out[f"{i}/cwrg_eg"] = numpy.asarray(eg, dtype=float64)
        out[f"{i}/cwrg_fg"] = numpy.asarray(fg, dtype=float64)
        out[f"{i}/cwrg_vg"] = numpy.asarray(vg, dtype=float64)

    return out


def basis_outputs(tmp_dir):
    """`eval_radial_basis` for each supported basis, over the same potential."""
    source = POTENTIAL.read_text()
    out = {}

    for basis in BASIS_TYPES:
        path = tmp_dir / f"{basis}.mtp"
        path.write_text(source.replace("radial_basis_type = RBChebyshev\n", f"radial_basis_type = {basis}\n", 1))
        pot = MTPTraining(str(path))
        if pot.get_radial_basis_type() != basis:
            raise RuntimeError(f"{basis} did not survive the rewrite; write_mtp or read_file has regressed")
        min_dist = pot.get_min_cutoff()
        for dist in PROBE_DISTANCES + [min_dist * 0.5]:
            vals, ders = pot.eval_radial_basis(float(dist))
            out[f"basis/{basis}/{dist:.6f}/vals"] = numpy.asarray(vals, dtype=float64)
            out[f"basis/{basis}/{dist:.6f}/ders"] = numpy.asarray(ders, dtype=float64)

    return out


def collect(tmp_dir):
    pot = MTPTraining(str(POTENTIAL))
    collected = outputs(pot, pot.get_max_cutoff())
    collected.update(basis_outputs(tmp_dir))
    return collected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true", help="rewrite the golden values from this build")
    parser.add_argument("--tol", type=float, default=TOL, help=f"relative tolerance (default {TOL:g})")
    args = parser.parse_args()

    tmp_dir = REFERENCE / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    collected = collect(tmp_dir)
    for stale in tmp_dir.glob("*.mtp"):
        stale.unlink()
    tmp_dir.rmdir()

    if args.capture:
        numpy.savez_compressed(GOLDEN, **collected)
        print(f"captured {len(collected)} arrays to {GOLDEN} ({GOLDEN.stat().st_size / 1024:.0f} KB)")
        return 0

    golden = numpy.load(GOLDEN)
    failures, worst = [], 0.0

    for name in sorted(golden.files):
        if name not in collected:
            failures += [f"{name}: missing from this build"]
            continue
        want, got = golden[name], collected[name]
        if got.shape != want.shape:
            failures += [f"{name}: shape {got.shape} != {want.shape}"]
            continue
        rel = float(numpy.abs(got - want).max()) / max(float(numpy.abs(want).max()), 1e-300)
        worst = max(worst, rel)
        if rel > args.tol:
            failures += [f"{name}: rel {rel:.3e}"]

    for name in sorted(set(collected) - set(golden.files)):
        failures += [f"{name}: present in this build, absent from the golden values"]

    print(f"{len(golden.files)} arrays checked, worst relative difference {worst:.3e}, tolerance {args.tol:g}")
    for line in failures[:20]:
        print(f"  FAIL {line}")
    if len(failures) > 20:
        print(f"  ... and {len(failures) - 20} more")
    print("all kernels match" if not failures else f"{len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
