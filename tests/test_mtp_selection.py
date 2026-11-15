"""The MTP's active learning against mlp's, in nbh and cfg modes, on the 2-species potential and the synthetic structures.

Under --recapture, mlp builds the active set of the structures (`mlp train --iteration_limit=0`), grades the candidates
against it (`mlp calculate_grade`) and selects among them (`mlp select_add`); mtp_reference/selection.npz stores its
active rows, log |det A|, grades and selection.
"""

import shutil
import subprocess

import numpy
import pytest

from conftest import DATA, MTP_REFERENCE, StoredOutputs, two_species
from otf_engine.almtp_io import read_mvs_header, read_mvs_state, write_mvs_state
from otf_engine.io_cfg import read_cfg, write_cfg
from otf_engine.selection import calculate_grade, select_add, update_active_set

POTENTIAL = DATA / "two-species-08.mtp"
MODES = ["nbh", "cfg"]


def fingerprint(atoms):
    return (numpy.round(atoms.cell[:], 4) + 0.0).tobytes(), (numpy.round(atoms.positions, 4) + 0.0).tobytes()


def active_rows(path, training):
    """The (structure, equation) of each set row of the active set in path, the structure an index into training, and log |det A|."""
    _, active_set, stored = read_mvs_state(str(path))
    index_of = {fingerprint(atoms): k for k, atoms in enumerate(training)}
    structure_of = [index_of[fingerprint(atoms)] for atoms in stored]
    rows = sorted((structure_of[c], int(e)) for c, e in zip(active_set.active_struct_indices.tolist(), active_set.active_eqn_indices.tolist(), strict=True) if c >= 0)
    log_det = numpy.linalg.slogdet(numpy.asarray(active_set.A))[1]
    return numpy.array(rows, dtype=float).reshape(-1, 2), log_det


def unlabelled(structures):
    """Copies without the grades a previous grading left."""
    out = []
    for atoms in structures:
        copy = atoms.copy()
        copy.calc = atoms.calc
        copy.arrays.pop("nbh_grades", None)
        copy.info.pop("features", None)
        out += [copy]
    return out


@pytest.fixture(scope="module")
def stored(recapture):
    outputs = StoredOutputs(MTP_REFERENCE / "selection.npz", recapture)
    yield outputs
    outputs.save()


@pytest.fixture(scope="module")
def sets(structures, candidates):
    return two_species(structures), two_species(candidates)


def capture(mode, training, cands, stored, mlip3, work):
    mlp = str(mlip3 / "bin" / "mlp")
    with open(work / "train.cfg", "w") as f:
        write_cfg(f, unlabelled(training))
    with open(work / "cands.cfg", "w") as f:
        write_cfg(f, unlabelled(cands))
    shutil.copy(POTENTIAL, work / "pot.almtp")
    subprocess.run([mlp, "train", "pot.almtp", "train.cfg", "--iteration_limit=0", "--no_mindist_update=true", f"--al_mode={mode}", "--save_to=mlp.almtp"], check=True, capture_output=True, cwd=work)
    rows, log_det = active_rows(work / "mlp.almtp", training)
    stored[f"{mode}/rows"], stored[f"{mode}/log_det"] = rows, numpy.array([log_det])

    subprocess.run([mlp, "calculate_grade", "mlp.almtp", "cands.cfg", "graded.cfg"], check=True, capture_output=True, cwd=work)
    with open(work / "graded.cfg.0") as f:
        graded = read_cfg(f)
    stored[f"{mode}/grades"] = numpy.array([float(atoms.info["features"]["MV_grade"]) for atoms in graded])
    if mode == "nbh":
        stored[f"{mode}/nbh_grades"] = numpy.concatenate([atoms.arrays["nbh_grades"] for atoms in graded])

    subprocess.run([mlp, "select_add", "mlp.almtp", "train.cfg", "cands.cfg", "selected.cfg"], check=True, capture_output=True, cwd=work)
    with open(work / "selected.cfg") as f:
        selected = read_cfg(f) if (work / "selected.cfg").stat().st_size else []
    index_of = {fingerprint(atoms): k for k, atoms in enumerate(cands)}
    stored[f"{mode}/selected"] = numpy.array(sorted(index_of[fingerprint(atoms)] for atoms in selected), dtype=float)


@pytest.mark.parametrize("mode", MODES)
def test_against_mlp(mode, sets, stored, recapture, mlip3, tmp_path):
    """update_active_set picks mlp's active rows, calculate_grade gives mlp's grades, and select_add selects mlp's candidates."""
    training, cands = sets
    if recapture:
        (tmp_path / "mlp").mkdir()
        capture(mode, training, cands, stored, mlip3, tmp_path / "mlp")

    potential = tmp_path / "pot.almtp"
    shutil.copy(POTENTIAL, potential)
    update_active_set(str(potential), unlabelled(training), al_mode=mode)
    rows, log_det = active_rows(potential, training)
    numpy.testing.assert_array_equal(rows, stored[f"{mode}/rows"])
    assert abs(log_det - stored[f"{mode}/log_det"][0]) <= 1e-10 * abs(log_det)

    # mlp writes a grade to 5 decimals or 7 significant digits
    graded = calculate_grade(str(potential), unlabelled(cands))
    grades = numpy.array([atoms.info["features"]["MV_grade"] for atoms in graded])
    numpy.testing.assert_allclose(grades, stored[f"{mode}/grades"], rtol=1e-6, atol=5e-6)
    if mode == "nbh":
        numpy.testing.assert_allclose(numpy.concatenate([atoms.arrays["nbh_grades"] for atoms in graded]), stored[f"{mode}/nbh_grades"], rtol=1e-6, atol=5e-6)

    selected, _ = select_add(str(potential), unlabelled(training), unlabelled(cands))
    index_of = {fingerprint(atoms): k for k, atoms in enumerate(cands)}
    assert sorted(index_of[fingerprint(atoms)] for atoms in selected) == stored[f"{mode}/selected"].astype(int).tolist()


def test_extrapolation_grades(sets, tmp_path):
    """PairMTPExtrapolation grades a structure as calculate_grade does, against the same active set."""
    from otf_engine._core.neighbors import neighbors
    from otf_engine._mtp import PairMTPExtrapolation

    training, cands = sets
    potential = tmp_path / "pot.almtp"
    shutil.copy(POTENTIAL, potential)
    update_active_set(str(potential), unlabelled(training), al_mode="nbh")
    graded = calculate_grade(str(potential), unlabelled(cands))

    _, active_set = read_mvs_header(str(potential))
    grader = PairMTPExtrapolation(str(POTENTIAL))
    grader.set_active_set(numpy.asarray(active_set.invA), False, 2)
    for atoms in graded:
        result = grader.compute(neighbors(atoms, grader.get_max_cutoff()))
        assert abs(result["max_grade"] - atoms.info["features"]["MV_grade"]) <= 1e-12 * atoms.info["features"]["MV_grade"]
        numpy.testing.assert_allclose(result["nbh_grades"], atoms.arrays["nbh_grades"], rtol=1e-12)


def test_mvs_round_trip(sets, tmp_path):
    """write_mvs_state then read_mvs_state gives back the weights, A, invA, the active rows and the structures they come from."""
    training, _ = sets
    potential = tmp_path / "pot.almtp"
    shutil.copy(POTENTIAL, potential)
    update_active_set(str(potential), unlabelled(training), al_mode="nbh")
    weights, active_set, stored = read_mvs_state(str(potential))

    again = tmp_path / "again.almtp"
    shutil.copy(POTENTIAL, again)
    write_mvs_state(str(again), weights, active_set, stored, 0)
    weights2, active_set2, stored2 = read_mvs_state(str(again))
    assert weights2 == weights
    numpy.testing.assert_array_equal(active_set2.A, active_set.A)
    numpy.testing.assert_array_equal(active_set2.invA, active_set.invA)
    numpy.testing.assert_array_equal(active_set2.active_struct_indices, active_set.active_struct_indices)
    numpy.testing.assert_array_equal(active_set2.active_eqn_indices, active_set.active_eqn_indices)
    assert [fingerprint(a) for a in stored2] == [fingerprint(a) for a in stored]
