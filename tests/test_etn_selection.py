"""The ETN's active learning in otf_engine.selection, which follows the MTP's rules, and PairETNExtrapolation's grades."""

import shutil

import numpy
import pytest
from ase.calculators.singlepoint import SinglePointCalculator

from etn_potentials import reference_potential, untrained_etn
from otf_engine._core import PairCalculator, neighbors
from otf_engine.selection import calculate_grade, update_active_set
from otf_engine.train import train

POTENTIAL = "etn-3"
MODES = ["nbh", "cfg"]
THRESHOLD = 1.001


def test_train_writes_active_set(structures, tmp_path):
    """train on an untrained .etn writes a trained potential whose active set grades the training structures below the selection threshold."""
    from otf_engine._etn import PairETN

    untrained = tmp_path / f"{POTENTIAL}-template.etn"
    untrained_etn(POTENTIAL).write(untrained)
    trained = str(tmp_path / f"{POTENTIAL}-trained.etn")
    # references to fit: the stored potential's energies and forces
    reference = PairCalculator(PairETN(str(reference_potential(POTENTIAL))))
    labelled = []
    for atoms in structures[:4]:
        reference.calculate(atoms, ["energy", "forces"])
        labelled += [atoms.copy()]
        labelled[-1].calc = SinglePointCalculator(labelled[-1], energy=reference.results["energy"], forces=reference.results["forces"])
    train(str(untrained), labelled, trained, options={"iteration_limit": 3, "skip_preinit": True, "log": "none"}, ranks=2)

    assert PairETN(trained).is_trained()
    grades = [atoms.info["features"]["MV_grade"] for atoms in calculate_grade(trained, [atoms.copy() for atoms in labelled])]
    assert max(grades) <= THRESHOLD


def test_force_weights_raise(structures, tmp_path):
    """Selection weights on forces raise for an ETN, which has no force gradients."""
    path = str(tmp_path / f"{POTENTIAL}.etn")
    shutil.copy(reference_potential(POTENTIAL), path)
    with pytest.raises(RuntimeError):
        update_active_set(path, structures, weights={"energy_weight": 1.0, "force_weight": 1.0, "stress_weight": 0.0, "site_en_weight": 0.0, "weight_scaling": 2})


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", ["etn-3", "etn-5"])
def test_extrapolation_grades(name, mode, structures, candidates, tmp_path):
    """PairETNExtrapolation grades the candidates as calculate_grade does against the same active set, and gives
    PairETN's energy, forces and virial; etn-5's middle cores have imaginary parts."""
    from otf_engine._etn import PairETN, PairETNExtrapolation
    from otf_engine.io_mvs import read_mvs_header

    path = str(tmp_path / f"{name}.etn")
    shutil.copy(reference_potential(name), path)
    update_active_set(path, structures, al_mode=mode)
    graded = calculate_grade(path, [atoms.copy() for atoms in candidates])

    weights, active_set = read_mvs_header(path)
    grader = PairETNExtrapolation(path)
    grader.set_active_set(numpy.asarray(active_set.invA), mode == "cfg", int(weights["weight_scaling"]))
    plain = PairETN(path)
    for atoms in graded:
        nl = neighbors(atoms, grader.get_max_cutoff())
        result, expected = grader.compute(nl), plain.compute(nl)
        assert abs(result["max_grade"] - atoms.info["features"]["MV_grade"]) <= 1e-10 * atoms.info["features"]["MV_grade"]
        if mode == "nbh":
            numpy.testing.assert_allclose(result["nbh_grades"], atoms.arrays["nbh_grades"], rtol=1e-10)
        for key in ("energy", "eatom", "forces", "virial"):
            numpy.testing.assert_allclose(result[key], expected[key], rtol=1e-12, atol=1e-12)
