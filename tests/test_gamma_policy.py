"""OTFCycle.preselect's gamma policy and its state in otf_state.json, on graded structures whose grades the test sets."""

import json
import shutil

import numpy
import pytest
from ase import Atoms

from conftest import DATA, two_species
from otf_engine import NestedLauncher
from otf_engine.otf_mtp import OTFCycle


@pytest.fixture
def workdir(structures, tmp_path, monkeypatch):
    """A directory holding a 2-species potential with an active set and its training set, the cycle's working directory."""
    from otf_engine.io_cfg import write_cfg
    from otf_engine.selection import update_active_set

    monkeypatch.chdir(tmp_path)
    training = two_species(structures)
    with open("train.cfg", "w") as f:
        write_cfg(f, training)
    shutil.copy(DATA / "two-species-08.mtp", "potential.almtp")
    update_active_set("potential.almtp", training)
    return tmp_path


def graded(*gammas):
    """One dump of single-atom structures graded gammas, each atom placed at its gamma."""
    structures = []
    for g in gammas:
        atoms = Atoms("H", positions=[[g, 0.0, 0.0]], cell=[10.0, 10.0, 10.0], pbc=True)
        atoms.info["features"] = {"MV_grade": float(g)}
        structures += [atoms]
    return {"dump.lammps": structures}


def preselected(state=None, gammas=(), **options):
    """The gammas preselect keeps, the cycle's branch, and the state it leaves in otf_state.json."""
    if state is not None:
        with open("otf_state.json", "w") as f:
            json.dump(state, f)
    with OTFCycle([], NestedLauncher(), **options) as cycle:
        kept = sorted(atoms.info["features"]["MV_grade"] for atoms in cycle.preselect(graded(*gammas)))
        branch = cycle.record["selection_branch"]
    with open("otf_state.json") as f:
        saved = json.load(f)
    return kept, branch, saved


def test_normal(workdir):
    """Of the candidates above gamma_tolerance, those below gamma_max."""
    kept, branch, state = preselected(gammas=(0.5, 2.0, 3.0, 7.0), gamma_max=5.0)
    assert (kept, branch) == ([2.0, 3.0], "normal")
    assert state["consecutive_non_extreme"] == 1
    assert "gamma_max0" not in state    # a candidate below gamma_max leaves gamma_max0 as it is


def test_intermediate(workdir):
    """With none below gamma_max, the lowest when it lies below gamma_max0, which starts at gamma_max_cap and becomes the
    mean of the lowest gammas."""
    kept, branch, state = preselected(gammas=(0.5, 3.0, 2.0, 7.0))
    assert (kept, branch) == ([2.0], "intermediate")
    assert state["gamma_max0"] == 2.0
    assert state["gamma_max0_history"] == [2.0]


def test_extreme(workdir):
    """With the lowest above gamma_max0, the lowest while extremes are allowed, which resets the run of non-extreme cycles."""
    kept, branch, state = preselected({"gamma_max0": 1.5, "consecutive_non_extreme": 3}, gammas=(3.0, 2.0))
    assert (kept, branch) == ([2.0], "extreme")
    assert state["consecutive_non_extreme"] == 0
    assert state["extreme_count"] == 1


def test_extremes_lock(workdir):
    """After extreme_lock_after_ntimes non-extreme cycles in a row, a cycle reaching the extreme branch selects nothing."""
    _, branch, state = preselected({"consecutive_non_extreme": 4}, gammas=(2.0,), gamma_max=5.0, extreme_lock_after_ntimes=5)
    assert branch == "normal" and state["extreme_allowed"] is False
    kept, branch, state = preselected(dict(state, gamma_max0=1.5), gammas=(3.0, 2.0), extreme_lock_after_ntimes=5)
    assert (kept, branch) == ([], "none")
    assert state["extreme_allowed"] is False


def test_gamma_max0_mean_of_last_ten(workdir):
    """gamma_max0 is the mean of the last 10 lowest gammas, never below gamma_max; a lowest gamma above gamma_max_cap is not recorded."""
    history = [float(g) for g in range(20, 31)]
    _, _, state = preselected({"gamma_max0_history": history[:10], "gamma_max0": 100.0}, gammas=(history[10],))
    assert state["gamma_max0_history"] == history[1:]
    assert state["gamma_max0"] == pytest.approx(numpy.mean(history[1:]))

    _, _, state = preselected(dict(state), gammas=(20000.0,))
    assert state["gamma_max0_history"] == history[1:]


def test_no_filtering_and_cap(workdir):
    """-P keeps every structure; max_structures then keeps that many at random."""
    kept, branch, _ = preselected(gammas=(0.5, 2.0, 3.0, 7.0), preselection_filtering=False)
    assert (kept, branch) == ([0.5, 2.0, 3.0, 7.0], "none")
    kept, _, _ = preselected(gammas=(0.5, 2.0, 3.0, 7.0), preselection_filtering=False, max_structures=2)
    assert len(kept) == 2 and set(kept) <= {0.5, 2.0, 3.0, 7.0}


def test_old_state_key(workdir):
    """A state saved under the old name non_extreme_count is read as consecutive_non_extreme."""
    _, _, state = preselected({"non_extreme_count": 2}, gammas=(2.0,), gamma_max=5.0)
    assert state["consecutive_non_extreme"] == 3
    assert "non_extreme_count" not in state


def test_cycle_recorded(workdir):
    """Leaving the block records the cycle in the state's history with its branch and archives it as ok."""
    _, _, state = preselected(gammas=(2.0,), gamma_max=5.0)
    entry = state["history"][-1]
    assert entry["selection_branch"] == "normal" and entry["n_preselected"] == 1
    assert (workdir / "otf_cycles" / "cycle_0" / "status").read_text().strip() == "ok"
