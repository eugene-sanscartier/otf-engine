"""otf_pool's operations on pyKMC's worker pool against the same steps in one process, and the cycle through them.

Under `mpirun -n 5`, pyKMC's ManagerFactory puts the manager on rank 0 and 4 workers in sessions of 2 ranks, with
otf_pool's operations as extra_ops (otf_pool_worker.py). On a dump of the candidates and the 2-species structures:

  update_active_set  as a global operation, leaves the active set of selection.update_active_set in this process: the
                     same active equations, and A to 1e-12 relatively
  grade_dump         submitted to a session, gives the grades of otf_cycle.grade_dump bit for bit
  select_add         as a global operation, offered the dump's highest-graded structures against the active set
                     update_active_set left, selects those selection.select_add selects in this process
  train_potential    as a global operation, logs the losses train.train logs on 4 threads to 1e-5 relatively, since under
                     varpro every loss goes through the linear solve, whose conditioning turns the reductions' order into
                     6e-7, and leaves the active set update_active_set finds on one rank from the same fit
  OTFCycle           stepped through as pyKMC steps it, archives a cycle as ok; given a missing dump, it raises on the
                     manager, archives the cycle as failed, and no rank hangs
"""

import json
import re
import shutil
import subprocess
import sys

import numpy
import pytest

from conftest import DATA, TESTS, two_species

pytest.importorskip("pykmc")

SPECIES = ["Ni", "Si"]
RANKS, SESSION_SIZE, ITERATIONS, CANDIDATES = 4, 2, 2, 4
LOSS_LINE = re.compile(r"(?:BFGS|Riemann) iter +(\d+): +(?:f=)?(\S+)")    # the trainer's logged losses


def write_dump(path, frames):
    """frames as a LAMMPS text dump, each cell turned into LAMMPS's triclinic orientation."""
    from ase.calculators.lammps import Prism

    lines = []
    for step, atoms in enumerate(frames):
        prism = Prism(atoms.cell.array)
        xhi, yhi, zhi, xy, xz, yz = (float(x) for x in prism.get_lammps_prism())
        positions = prism.vector_to_lammps(atoms.positions, wrap=True).tolist()
        lines += ["ITEM: TIMESTEP", str(step), "ITEM: NUMBER OF ATOMS", str(len(atoms)), "ITEM: BOX BOUNDS xy xz yz pp pp pp"]
        lines += [f"{min(0.0, xy, xz, xy + xz)!r} {xhi + max(0.0, xy, xz, xy + xz)!r} {xy!r}", f"{min(0.0, yz)!r} {yhi + max(0.0, yz)!r} {xz!r}", f"0.0 {zhi!r} {yz!r}"]
        lines += ["ITEM: ATOMS id type x y z"] + [f"{i + 1} {t + 1} {p[0]!r} {p[1]!r} {p[2]!r}" for i, (t, p) in enumerate(zip(atoms.arrays["type_index"], positions))]
    path.write_text("\n".join(lines) + "\n")


def same_active_set(mpi_set, serial_set):
    """Whether the workers' active set is the one-rank search's, each as read_mvs_state returns it: the same active
    equations, and A to 1e-12 relatively."""
    (_, mpi, _), (_, serial, _) = mpi_set, serial_set
    rows = lambda mv: [numpy.asarray(mv.active_pool_ids), numpy.asarray(mv.active_struct_indices), numpy.asarray(mv.active_eqn_indices)]
    assert all(numpy.array_equal(a, b) for a, b in zip(rows(mpi), rows(serial)))
    assert numpy.abs(mpi.A - serial.A).max() <= 1e-12 * numpy.abs(serial.A).max()


@pytest.mark.mpi
def test_otf_pool(structures, candidates, tmp_path):
    from otf_engine._mtp import PairMTPGrad, write_mtp
    from otf_engine.almtp_io import read_mvs_header, read_mvs_state
    from otf_engine.io_cfg import read_cfg, write_cfg
    from otf_engine.otf_cycle import grade_dump
    from otf_engine.selection import select_add, update_active_set
    from otf_engine.train import train

    # every input, written before mpirun starts the ranks
    work = tmp_path
    with open(work / "train.cfg", "w") as f:
        write_cfg(f, two_species(structures))
    shutil.copy(DATA / "two-species-08.mtp", work / "potential.almtp")
    with open(work / "train.cfg") as f:
        structs = read_cfg(f, SPECIES)
    update_active_set(str(work / "potential.almtp"), structs)
    write_dump(work / "dump.lammps", two_species(candidates))

    # the references, in this process
    shutil.copy(work / "potential.almtp", work / "serial_updated.almtp")
    update_active_set(str(work / "serial_updated.almtp"), structs)
    graded = grade_dump(str(work / "dump.lammps"), str(work / "potential.almtp"), SPECIES)
    grades = numpy.array([atoms.info["features"]["MV_grade"] for atoms in graded])
    offered = [graded[i] for i in numpy.argsort(grades)[::-1][:CANDIDATES]]
    selected = [next(i for i, c in enumerate(offered) if c is s) for s in select_add(str(work / "serial_updated.almtp"), structs, offered)[0]]
    train(str(work / "potential.almtp"), structs, str(work / "threads.almtp"), settings={"iteration_limit": ITERATIONS, "log": str(work / "threads_train.log")}, ranks=RANKS)

    cmd = ["mpirun", "-n", str(RANKS + 1), sys.executable, str(TESTS / "otf_pool_worker.py"), str(SESSION_SIZE), str(ITERATIONS), str(CANDIDATES), *SPECIES]
    proc = subprocess.run(cmd, cwd=work, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]

    numpy.testing.assert_array_equal(numpy.load(work / "mpi_grades.npy"), grades)
    same_active_set(read_mvs_state(str(work / "mpi_updated.almtp")), read_mvs_state(str(work / "serial_updated.almtp")))
    assert numpy.load(work / "mpi_selected.npy").tolist() == selected

    losses = {arm: numpy.array([float(m.group(2)) for m in LOSS_LINE.finditer((work / f"{arm}_train.log").read_text())]) for arm in ("threads", "mpi")}
    assert len(losses["mpi"]) == len(losses["threads"]) > 0
    numpy.testing.assert_allclose(losses["mpi"], losses["threads"], rtol=1e-5)

    # The fits differ in their linear coefficients, so the reference searches the MPI fit on one rank.
    write_mtp(PairMTPGrad(str(work / "mpi.almtp")), str(work / "serial.almtp"))
    update_active_set(str(work / "serial.almtp"), structs, weights=read_mvs_header(str(work / "potential.almtp"))[0])
    same_active_set(read_mvs_state(str(work / "mpi.almtp")), read_mvs_state(str(work / "serial.almtp")))

    outcomes = json.loads((work / "outcomes.json").read_text())
    assert outcomes == {"ok": "returned", "fail": "raised RuntimeError on missing.lammps"}
    assert [(work / f"otf_cycles/cycle_{k}/status").read_text().strip() for k in (0, 1)] == ["ok", "failed"]
