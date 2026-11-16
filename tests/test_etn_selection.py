"""The ETN's active learning in otf_engine.selection against mlip-4's CfgCalc, in nbh and cfg modes.

On the 3-core potential of test_etn_kernels, compares with mlip-4's CfgCalc, stored in etn_reference/selection.npz:

  active set   update_active_set from no active set on the structures: the structure and equation of each row of A, or an
               equal equation of another structure, which rounding may pick in its place, and log |det A|
  grades       calculate_grade on the candidates: each candidate's grade and, in nbh mode, its per-atom grades
  selection    select_add of the candidates: the selected ones, exactly, and log |det A| after

Grades and log det are compared to 1e-8 relatively: A's 1e-6 unset rows amplify rounding into 3e-9.
"""

import shutil
import subprocess
from pathlib import Path

import numpy
import pytest
from ase.calculators.singlepoint import SinglePointCalculator

from conftest import ETN_REFERENCE, StoredOutputs
from etn_potentials import compile_probe, params, reference_potential, untrained_etn
from otf_engine._core import PotentialCalculator, neighbors
from otf_engine.io_mvs import read_mvs_state
from otf_engine.selection import calculate_grade, select_add, update_active_set
from otf_engine.train import train

POTENTIAL = "etn-3"
MODES = ["nbh", "cfg"]
THRESHOLD_SWAP = 1.00001
EQUAL_EQUATIONS = 1e-12    # relative difference under which two equations are one
TOL = 1e-8


def write_selection_file(path, pot, mode, training, candidates):
    """mlip4_selection's input: shape and params of pot, the mode, the training structures and the candidates."""
    ints = lambda values: " ".join(map(str, values))
    floats = lambda values: " ".join(repr(float(x)) for x in values)
    lines = [f"{pot.get_species_count()} {len(pot.get_species_ranks())} {pot.get_core_count()} {pot.get_radial_basis_size()} {pot.get_min_cutoff()!r} {pot.get_max_cutoff()!r}"]
    lines += [ints(pot.get_species_ranks()), ints(pot.get_radial_func_counts()), ints(pot.get_radial_ranks())]
    lines += [f"{len(ranks)} {ints(ranks)}" for ranks in pot.get_bond_ranks()]
    coeffs = params(pot)
    lines += [str(len(coeffs)), floats(coeffs), str(int(mode == "nbh"))]
    for group in (training, candidates):
        lines += [str(len(group))]
        for atoms in group:
            lines += [str(len(atoms))] + [floats(row) for row in atoms.cell.array]
            lines += [f"{t} {floats(p)}" for t, p in zip(atoms.arrays["type_index"], atoms.positions)]
    Path(path).write_text("\n".join(lines) + "\n")


def parse_selection(printed):
    """The probe's "@ name ... : values" lines, as name -> list of floats; grades keyed "grade/c"."""
    out = {}
    for line in printed.splitlines():
        if not line.startswith("@ "): continue
        head, values = line[2:].split(" :") if " :" in line else (line[2:].rsplit(" ", 1)[0], line.rsplit(" ", 1)[1])
        words = head.split()
        key = f"grade/{words[1]}" if words[0] == "grade" else words[0]
        out[key] = [float(x) for x in values.split()]
    return out


def active_rows(path, training):
    """(structure, equation) of each row of A in the active set of the .etn at path, -1 -1 for an unset row."""
    _, active_set, stored = read_mvs_state(path)
    key = lambda atoms: numpy.round(atoms.positions, 6).tobytes()
    index_of = {key(atoms): s for s, atoms in enumerate(training)}
    labels = [index_of[key(cfg)] for cfg in stored]
    rows = numpy.array([[labels[c], e] if p == 0 else [-1, -1] for p, c, e in zip(active_set.active_pool_ids, active_set.active_struct_indices, active_set.active_eqn_indices)])
    return rows, active_set.A


def equation(pot, training, structure, eqn):
    """Equation eqn of a training structure: the energy over N (0) or a site energy's gradient (1 + 3 n + 9 + i)."""
    if structure < 0: return numpy.zeros(pot.get_coeff_count())
    atoms = training[structure]
    grads = pot.eval_grad(neighbors(atoms, pot.get_max_cutoff()))
    return grads.sum(axis=0) / len(atoms) if eqn == 0 else grads[eqn - (1 + 3 * len(atoms) + 9)]


def log_det(A):
    """log |det A| in base n, as mlip-4 keeps it."""
    return numpy.linalg.slogdet(A)[1] / numpy.log(len(A))


def relative(a, b):
    a, b = numpy.atleast_1d(numpy.asarray(a, dtype=float)), numpy.atleast_1d(numpy.asarray(b, dtype=float))
    return float(numpy.max(numpy.abs(a - b)) / max(numpy.max(numpy.abs(b)), 1e-300))


@pytest.fixture(scope="module")
def stored(recapture, mlip4_build, structures, candidates, tmp_path_factory):
    """mlip-4's selections, rerun first under --recapture."""
    from otf_engine._etn import PairETN

    outputs = StoredOutputs(ETN_REFERENCE / "selection.npz", recapture)
    if recapture:
        outputs.arrays.clear()
        workdir = tmp_path_factory.mktemp("etn_selection")
        probe = compile_probe(mlip4_build, workdir, ETN_REFERENCE / "mlip4_selection.cpp")
        pot = PairETN(str(reference_potential(POTENTIAL)))
        for mode in MODES:
            selection_file = workdir / f"{mode}.txt"
            write_selection_file(selection_file, pot, mode, structures, candidates)
            printed = subprocess.run([str(probe), str(selection_file)], check=True, capture_output=True, text=True).stdout
            for key, values in parse_selection(printed).items():
                outputs[f"{mode}/{key}"] = numpy.array(values)
        outputs.save()
    return outputs


@pytest.mark.parametrize("mode", MODES)
def test_against_mlip4(mode, structures, candidates, stored, tmp_path):
    from otf_engine._etn import ETN

    path = str(tmp_path / f"{POTENTIAL}.etn")
    shutil.copy(reference_potential(POTENTIAL), path)
    update_active_set(path, structures, al_mode=mode)
    rows, A = active_rows(path, structures)
    theirs = numpy.asarray(stored[f"{mode}/active_set"], dtype=int).reshape(-1, 2)
    # atoms of the same environment in two structures give one equation twice, and rounding picks either
    pot = ETN(path)
    mismatched = [k for k in numpy.nonzero(numpy.any(rows != theirs, axis=1))[0] if relative(equation(pot, structures, *rows[k]), equation(pot, structures, *theirs[k])) > EQUAL_EQUATIONS]
    assert not mismatched, f"{len(mismatched)} of {len(rows)} rows hold another equation than mlip-4's"
    assert relative(log_det(A), stored[f"{mode}/log_det"]) <= TOL

    graded = calculate_grade(path, [atoms.copy() for atoms in candidates])
    for c, atoms in enumerate(graded):
        theirs = stored[f"{mode}/grade/{c}"]
        assert relative(atoms.info["features"]["MV_grade"], theirs[0]) <= TOL, f"grade of candidate {c}"
        if mode == "nbh":
            assert relative(atoms.arrays["nbh_grades"], theirs[1:]) <= TOL, f"per-atom grades of candidate {c}"

    selected, after = select_add(path, structures, candidates)
    assert sorted(next(c for c, atoms in enumerate(candidates) if atoms is s) for s in selected) == sorted(int(c) for c in stored[f"{mode}/selected"])
    assert relative(log_det(after.A), stored[f"{mode}/log_det_after"]) <= TOL


def test_train_writes_active_set(structures, tmp_path):
    """train on an untrained .etn writes a trained potential whose active set grades the training structures below threshold_swap."""
    from otf_engine._etn import PairETN

    untrained = tmp_path / f"{POTENTIAL}-template.etn"
    untrained_etn(POTENTIAL).write(untrained)
    trained = str(tmp_path / f"{POTENTIAL}-trained.etn")
    # references to fit: the stored potential's energies and forces
    reference = PotentialCalculator(PairETN(str(reference_potential(POTENTIAL))))
    labelled = []
    for atoms in structures[:4]:
        reference.calculate(atoms, ["energy", "forces"])
        labelled += [atoms.copy()]
        labelled[-1].calc = SinglePointCalculator(labelled[-1], energy=reference.results["energy"], forces=reference.results["forces"])
    train(str(untrained), labelled, trained, settings={"iteration_limit": 3, "skip_preinit": True, "log": "none"}, ranks=2)

    assert PairETN(trained).is_trained()
    grades = [atoms.info["features"]["MV_grade"] for atoms in calculate_grade(trained, [atoms.copy() for atoms in labelled])]
    assert max(grades) <= THRESHOLD_SWAP


def test_force_weights_raise(structures, tmp_path):
    """Selection weights on forces raise for an ETN, which has no force gradients."""
    path = str(tmp_path / f"{POTENTIAL}.etn")
    shutil.copy(reference_potential(POTENTIAL), path)
    with pytest.raises(RuntimeError):
        update_active_set(path, structures, weights={"energy_weight": 1.0, "force_weight": 1.0, "stress_weight": 0.0, "site_en_weight": 0.0, "weight_scaling": 2})
