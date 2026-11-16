"""The MTP's kernels: against mlip-3's evaluation, against each other, and their derivatives against finite differences.

The forward pass is written out five times, in PairMTP::compute, PairMTPExtrapolation::compute and MTP's
eval_basis, eval_grad and eval_loss_grad; each must give what the others give.
"""

import subprocess

import numpy
import pytest

from conftest import DATA, MTP_REFERENCE, StoredOutputs, parse_probe, two_species
from otf_engine._core import derivative_checks
from otf_engine._core.neighbors import neighbors

TEACHER = MTP_REFERENCE / "potential-16.mtp"
TWO_SPECIES = DATA / "two-species-08.mtp"
DISTANCES = [1.5, 2.5, 4.0, 4.99]


# name: (potential, whether the structures are renamed to 2 species)
POTENTIALS = {"level16": (TEACHER, False), "level8": (TWO_SPECIES, True)}


@pytest.fixture(scope="module")
def stored(recapture):
    outputs = StoredOutputs(MTP_REFERENCE / "kernels.npz", recapture)
    yield outputs
    outputs.save()


@pytest.mark.parametrize("name", POTENTIALS)
def test_against_mlp(name, structures, candidates, stored, mlip3_probe, tmp_path):
    """Energy, forces and virial match mlp's CalcEFS on every structure, and the radial basis its basis times the scaling."""
    from otf_engine._mtp import PairMTP
    from otf_engine.io_cfg import write_cfg

    potential, renamed = POTENTIALS[name]
    cells = structures + candidates
    if renamed: cells = two_species(cells)

    if stored.recapture:
        cfg = tmp_path / "cells.cfg"
        with open(cfg, "w") as f:
            write_cfg(f, cells)
        script = tmp_path / "script.txt"
        script.write_text("efs\n" + "".join(f"radial {r}\n" for r in DISTANCES))
        stored.replace_under(name, parse_probe(subprocess.run([str(mlip3_probe), str(potential), str(cfg), str(script)], check=True, capture_output=True, text=True).stdout))
    theirs = {}
    for key in stored.keys_under(name):
        theirs.setdefault(key.rsplit("/", 1)[1].split("_", 1)[1], []).append(stored[key])

    pot = PairMTP(str(potential))
    for k, atoms in enumerate(cells):
        result = pot.compute(neighbors(atoms, pot.get_max_cutoff()))
        assert abs(result["energy"] - theirs["energy"][k][0]) <= 1e-13 * abs(theirs["energy"][k][0])
        forces = theirs["forces"][k].reshape(-1, 3)
        assert numpy.max(numpy.abs(numpy.asarray(result["forces"]) - forces)) <= 1e-12 * numpy.max(numpy.abs(forces))
        stresses = theirs["stresses"][k].reshape(3, 3)
        virial = stresses[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]
        assert numpy.max(numpy.abs(numpy.asarray(result["virial"]) - virial)) <= 1e-12 * numpy.max(numpy.abs(virial))

    for k, r in enumerate(DISTANCES):
        vals, ders = pot.eval_radial_basis(r)
        numpy.testing.assert_allclose(vals, pot.get_scaling() * theirs["radial_vals"][k], rtol=1e-13, atol=1e-15)
        numpy.testing.assert_allclose(ders, pot.get_scaling() * theirs["radial_ders"][k], rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("name", POTENTIALS)
def test_forward_passes_agree(name, structures):
    """The five forward passes give one energy, forces and virial: compute's, the extrapolation's, and the linear block
    of eval_grad and eval_basis contracted with the coefficients; eval_loss_grad is eval_grad contracted with the loss's
    derivatives."""
    from otf_engine._mtp import MTP, PairMTPExtrapolation

    potential, renamed = POTENTIALS[name]
    cells = two_species(structures) if renamed else structures
    pot = MTP(str(potential))
    grader = PairMTPExtrapolation(str(potential))
    n = pot.get_coeff_count()
    grader.set_active_set(numpy.random.default_rng(1).standard_normal((n, n)), False, 2)
    coeffs = numpy.asarray(pot.get_coeffs())
    lin = slice(pot.get_radial_coeff_count(), n)
    first_linear = pot.get_radial_coeff_count() + pot.get_species_count()

    for atoms in cells:
        nl = neighbors(atoms, pot.get_max_cutoff())
        result = pot.compute(nl)
        energy, forces, virial = result["energy"], numpy.asarray(result["forces"]), numpy.asarray(result["virial"])
        scale_f, scale_v = numpy.max(numpy.abs(forces)), numpy.max(numpy.abs(virial))

        graded = grader.compute(nl)
        assert numpy.max(numpy.abs(numpy.asarray(graded["forces"]) - forces)) <= 1e-13 * scale_f
        assert numpy.max(numpy.abs(numpy.asarray(graded["virial"]) - virial)) <= 1e-13 * scale_v
        assert abs(numpy.sum(result["eatom"]) - energy) <= 1e-13 * abs(energy)

        site, force_grad, virial_grad = (numpy.asarray(x) for x in pot.eval_grad(nl, forces=True, virial=True))
        assert abs(site[:, lin].sum(axis=0) @ coeffs[lin] - energy) <= 1e-13 * abs(energy)
        assert numpy.max(numpy.abs(force_grad[:, :, lin] @ coeffs[lin] - forces)) <= 1e-12 * scale_f
        assert numpy.max(numpy.abs(virial_grad[:, lin] @ coeffs[lin] - virial)) <= 1e-12 * scale_v
        numpy.testing.assert_array_equal(numpy.asarray(pot.eval_grad(nl)), site)
        numpy.testing.assert_allclose(numpy.asarray(pot.eval_basis(nl)), site[:, first_linear:], rtol=0, atol=1e-14 * numpy.max(numpy.abs(site[:, first_linear:])))

        linear_only = [numpy.asarray(x) for x in pot.eval_grad(nl, forces=True, virial=True, radial=False)]
        for full, part in zip((site, force_grad, virial_grad), linear_only):
            numpy.testing.assert_array_equal(part[..., lin], full[..., lin])

        rng = numpy.random.default_rng(2)
        dloss_denergy, dloss_dforces, dloss_dvirial = float(rng.standard_normal()), rng.standard_normal(forces.shape), rng.standard_normal(6)
        contracted = dloss_denergy * site.sum(axis=0) + numpy.einsum("iac,ia->c", force_grad, dloss_dforces) + dloss_dvirial @ virial_grad
        loss_grad = numpy.asarray(pot.eval_loss_grad(nl, dloss_denergy, dloss_dforces, dloss_dvirial))
        assert numpy.max(numpy.abs(loss_grad - contracted)) <= 1e-12 * numpy.max(numpy.abs(contracted))


@pytest.mark.parametrize("name", POTENTIALS)
def test_derivatives(name, structures):
    """Forces, virial and every coefficient gradient match central differences; the energy is invariant under rotation and reflection."""
    from otf_engine._mtp import MTP

    potential, renamed = POTENTIALS[name]
    atoms = (two_species(structures) if renamed else structures)[1]
    pot = MTP(str(potential))
    derivative_checks.check_gradients(pot, atoms)
    assert derivative_checks.check_virial(pot, atoms) < 1e-6
    assert derivative_checks.check_invariance(pot, atoms) < 1e-12
