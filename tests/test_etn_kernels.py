"""The ETN's kernels against mlip-4's, and their derivatives against finite differences.

Every tensor PairETN computes is compared with mlip-4's, stored in etn_reference/kernels.npz: the radial basis and its
derivatives, the real spherical harmonics and their gradients, the dense real and imaginary 3-j tables, and, for six
potentials on three cells, the energy, forces and virial, each atom's site energy, its input and bond vectors, the gradient
of a loss with seeded derivatives, and the first atoms' site-energy gradients. The potentials span 1 to 5 cores, l up to 4, bonds with
more l than the input, zero ranks and a single species; the cells a synthetic structure, an atom alone with its images
and a pair. Each array is compared relative to its largest entry, and each row, at a looser tolerance, relative to its own.
"""

import subprocess

import numpy
import pytest
from ase import Atoms

from conftest import ETN_REFERENCE, StoredOutputs
from etn_potentials import MAX_DIST, MIN_DIST, POTENTIALS, RADIAL_BASIS_SIZE, coeff_blocks, compile_probe, get_block, params, parse_tensors, reference_potential, set_params, species_of, untrained_etn
from otf_engine._core import neighbors
from otf_engine._core.derivative_checks import check_forces, check_invariance, check_virial

TOL = 1e-12
ROW_TOL = 1e-9    # rows whose terms cancel, such as small net forces, round to 1e-11 of themselves
PROBE_DISTANCES = [1.0, 1.5, 2.5, 4.0, 4.9]
WIGNER_L_COUNT = 5
HARMONICS_L_COUNT = 5
DIRECTIONS = numpy.array([[0.3, -0.5, 0.7], [0.0, 0.0, 1.0], [-1.0, 2.0, 0.5], [0.6, 0.8, 0.0]])
GRAD_ATOMS = 3    # the atoms whose site-energy gradients are compared


def directions():
    return DIRECTIONS / numpy.linalg.norm(DIRECTIONS, axis=1)[:, None]


@pytest.fixture(scope="module")
def cells(structures):
    """Structure 1, an atom alone with its images 3.5 A away, and a pair 3.7 A apart, with their species."""
    alone = Atoms(numbers=[1], positions=[[0.0, 0.0, 0.0]], cell=3.5 * numpy.eye(3), pbc=True)
    pair = Atoms(numbers=[1, 2], positions=[[1.0, 1.0, 1.0], [4.7, 1.0, 1.0]], cell=9.0 * numpy.eye(3), pbc=True)
    built = [(structures[k], numpy.asarray(structures[k].arrays["type_index"], dtype=numpy.int32)) for k in (1,)]
    return built + [(alone, numpy.array([0], dtype=numpy.int32)), (pair, numpy.array([0, 1], dtype=numpy.int32))]


def loss_seeds(s, n_atoms):
    """Seeded derivatives of a loss with respect to structure s's energy, virial (xx, yy, zz, xy, xz, yz) and forces."""
    rng = numpy.random.default_rng(100 + s)
    return float(rng.standard_normal()), rng.standard_normal(6), rng.standard_normal((n_atoms, 3))


def outputs(potentials, cells):
    """Every tensor of this build, keyed as mlip-4's."""
    from otf_engine._etn import real_harmonics, real_wigner_3j

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
                out[prefix + key] = numpy.asarray(result[key], dtype=numpy.float64)
            out[prefix + "inputs"] = numpy.asarray(result["inputs"])
            for b, (re_part, im_part) in enumerate(zip(result["bonds"], result["bonds_im"])):
                out[prefix + f"bonds/{b}"] = numpy.asarray(re_part)
                out[prefix + f"bonds_im/{b}"] = numpy.asarray(im_part)
            dloss_denergy, dloss_dvirial, dloss_dforces = loss_seeds(s, len(atoms))
            out[prefix + "loss_grad"] = numpy.asarray(pot.eval_loss_grad(nl, dloss_denergy, dloss_dforces, dloss_dvirial))
            out[prefix + "grad"] = numpy.asarray(pot.eval_grad(nl))[:GRAD_ATOMS]
    return out


def run_probe(probe, pot, cells, workdir):
    """mlip-4's tensors for pot on cells."""
    ints = lambda values: " ".join(map(str, values))
    floats = lambda values: " ".join(repr(float(x)) for x in values)
    lines = [f"{pot.get_species_count()} {len(pot.get_species_ranks())} {pot.get_core_count()} {RADIAL_BASIS_SIZE} {MIN_DIST!r} {MAX_DIST!r}"]
    lines += [ints(pot.get_species_ranks()), ints(pot.get_radial_func_counts()), ints(pot.get_radial_ranks())]
    lines += [f"{len(ranks)} {ints(ranks)}" for ranks in pot.get_bond_ranks()]
    coeffs = params(pot)
    lines += [str(len(coeffs)), floats(coeffs)]
    lines += [str(len(PROBE_DISTANCES)), floats(PROBE_DISTANCES), str(WIGNER_L_COUNT)]
    lines += [f"{HARMONICS_L_COUNT} {len(DIRECTIONS)}"] + [floats(u) for u in directions()]
    most_atoms = max(len(atoms) for atoms, _ in cells)
    lines += [f"{most_atoms} {GRAD_ATOMS} {len(cells)}"]
    for s, (atoms, types) in enumerate(cells):
        lines += [str(len(atoms))] + [floats(row) for row in atoms.cell.array]
        lines += [f"{t} {floats(p)}" for t, p in zip(species_of(pot, types), atoms.positions)]
        dloss_denergy, dloss_dvirial, dloss_dforces = loss_seeds(s, len(atoms))
        lines += [floats([dloss_denergy]), floats(dloss_dvirial)] + [floats(row) for row in dloss_dforces]
    probe_file = workdir / "probe.txt"
    probe_file.write_text("\n".join(lines) + "\n")
    return parse_tensors(subprocess.run([str(probe), str(probe_file)], check=True, capture_output=True, text=True).stdout)


@pytest.fixture(scope="module")
def potentials(recapture, mlip4_build, cells, tmp_path_factory):
    """The test potentials, ETN's of etn_reference/; under --recapture built first from their shapes with
    seeded coefficients, and mlip-4's tensors stored."""
    from otf_engine._etn import ETN, write_etn

    if recapture:
        stored = StoredOutputs(ETN_REFERENCE / "kernels.npz", True)
        stored.arrays.clear()
        workdir = tmp_path_factory.mktemp("etn_kernels")
        probe = compile_probe(mlip4_build, workdir, ETN_REFERENCE / "mlip4_probe.cpp")
        for k, name in enumerate(POTENTIALS):
            pot = untrained_etn(name)
            set_params(pot, numpy.random.default_rng(k).standard_normal(pot.get_coeff_count()) * 0.3)
            pot.set_trained(True)
            write_etn(pot, reference_potential(name))
            for key, value in run_probe(probe, pot, cells, workdir).items():
                stored[f"{name}/{key}" if key.startswith("s") else key] = value
        stored.save()
    return {name: ETN(str(reference_potential(name))) for name in POTENTIALS}


def relative_differences(got, want):
    """Largest difference relative to the array's largest entry, and relative to each row's, floored at 1e-6 of the array's."""
    got, want = numpy.atleast_2d(got), numpy.atleast_2d(want)
    if got.ndim > 2:
        got, want = got.reshape(len(got), -1), want.reshape(len(want), -1)
    array_max = max(float(numpy.abs(want).max(initial=0.0)), 1e-300)
    diff = numpy.abs(got - want)
    scale = numpy.maximum(numpy.abs(want).max(axis=1, initial=0.0), 1e-6 * array_max)
    return float(diff.max(initial=0.0)) / array_max, float((diff.max(axis=1, initial=0.0) / scale).max(initial=0.0))


def test_against_mlip4(potentials, cells):
    golden = numpy.load(ETN_REFERENCE / "kernels.npz")
    collected = outputs(potentials, cells)
    assert sorted(collected) == sorted(golden.files)
    failures = []
    for name in sorted(collected):
        want, got = golden[name], collected[name]
        assert got.shape == want.shape, f"{name}: shape {got.shape} vs mlip-4's {want.shape}"
        if numpy.abs(want).max(initial=0.0) < 1e-10:    # zero by symmetry, as the forces on an atom alone with its images
            if numpy.abs(got).max(initial=0.0) >= 1e-10:
                failures += [f"{name}: {numpy.abs(got).max():.1e} where mlip-4's vanishes"]
            continue
        rel, rel_row = relative_differences(got, want)
        if rel > TOL or rel_row > ROW_TOL:
            failures += [f"{name}: {rel:.1e}, by row {rel_row:.1e}"]
    assert not failures, failures


@pytest.mark.parametrize("name", ["etn-4", "etn-5"])
def test_derivatives(name, potentials, cells):
    """Forces and virial match central differences, and the energy is invariant under rotation and reflection."""
    atoms, types = cells[0]
    atoms = atoms.copy()
    pot = potentials[name]
    atoms.arrays["type_index"] = species_of(pot, types)
    assert check_forces(pot, atoms) < 1e-6
    assert check_virial(pot, atoms) < 1e-6
    assert check_invariance(pot, atoms) < 1e-12


def test_loss_grad(potentials, cells, step=1e-5):
    """eval_loss_grad matches central differences along a random direction in each coefficient block of the 3-core potential.

    The imaginary cores' derivatives of the 5-core potential are 1e-7 of the loss, below what differences resolve.
    """
    pot = potentials["etn-3"]
    atoms, types = cells[0]
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
    offset = 0
    for kind, arg in coeff_blocks(pot):
        size = len(get_block(pot, kind, arg))
        direction = numpy.zeros_like(coeffs)
        direction[offset : offset + size] = rng.standard_normal(size)
        offset += size
        analytic = grad @ direction
        if size == 0 or analytic == 0.0:
            continue
        numeric = (loss(coeffs + step * direction) - loss(coeffs - step * direction)) / (2 * step)
        assert abs(numeric - analytic) <= 1e-6 * abs(analytic), f"{kind} {arg}"
    pot.set_coeffs(coeffs)


@pytest.mark.parametrize("name", POTENTIALS)
def test_round_trip(name, potentials, tmp_path):
    """write_etn then PairETN gives back every coefficient of a trained potential, which stays trained."""
    from otf_engine._etn import PairETN, write_etn

    write_etn(potentials[name], tmp_path / "back.etn")
    back = PairETN(str(tmp_path / "back.etn"))
    numpy.testing.assert_array_equal(params(back), params(potentials[name]))
    assert back.is_trained()


@pytest.mark.parametrize("species_count", [3, 0], ids=["species_count", "no_species_count"])
def test_untrained(species_count, tmp_path):
    """An ETN built from its shape starts untrained from mlip-4's sin(10 + k^2), and write_etn writes a file, with or
    without species_count, that PairETN reads back to the same shape and coefficients, still untrained."""
    from otf_engine._etn import PairETN, write_etn

    built = untrained_etn("etn-5", species_count)
    initial = numpy.sin(10 + numpy.arange(built.get_coeff_count()) ** 2)
    numpy.testing.assert_array_equal(params(built), initial)
    assert not built.is_trained()

    write_etn(built, tmp_path / "untrained.etn")
    assert ("species_count" in (tmp_path / "untrained.etn").read_text()) == (species_count > 0)
    back = PairETN(str(tmp_path / "untrained.etn"))
    numpy.testing.assert_array_equal(params(back), initial)
    assert not back.is_trained()
    assert back.get_species_count() == species_count
    for get in ("get_species_ranks", "get_radial_func_counts", "get_radial_ranks"):
        numpy.testing.assert_array_equal(getattr(back, get)(), getattr(built, get)(), err_msg=get)
    assert [list(r) for r in back.get_bond_ranks()] == [list(r) for r in built.get_bond_ranks()]
