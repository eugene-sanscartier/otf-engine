import concurrent.futures
import datetime
import json
import logging
import os
import shutil
import threading
import traceback
from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass
from pathlib import Path

import numpy

import ase
import ase.io.lammpsrun

from .io_cfg import read_cfg, write_cfg
from .mtp_backend import calculate_grade, errors, select_add, update_active_set
from .almtp_io import read_mvs_state
from .cycles import CYCLE_PREFIX, LOG_FILE, LOG_FORMAT, TRAIN_LOG, archive_cycle, next_cycle_dir
from .launchers import Launcher, JobTimedOut, JobOutOfMemory

logger = logging.getLogger(__name__)

OTF_STATE_FILE = "otf_state.json"
DEFERRED_EVALS_FILE = "deferred_evals.extxyz"  # beside the cycle directories: the evaluations a cycle resubmits from the one before
_STATE_LOCK = threading.Lock()


def _load_state():
    if os.path.isfile(OTF_STATE_FILE):
        with open(OTF_STATE_FILE, "r") as f:
            state = json.load(f)
    else:
        state = {}
    if "non_extreme_count" in state and "consecutive_non_extreme" not in state:
        state["consecutive_non_extreme"] = state.pop("non_extreme_count")
    return state


def _save_state(state):
    """Replace OTF_STATE_FILE with state, one writer at a time: the launcher saves from evaluation threads."""
    with _STATE_LOCK:
        with open(f"{OTF_STATE_FILE}.tmp", "w") as f:
            json.dump(state, f, indent=2)
        os.replace(f"{OTF_STATE_FILE}.tmp", OTF_STATE_FILE)


MAX_STRUCTURES_PER_DUMP = 10000


def grade_dump(dump, potential, species=None):
    """Parse one extrapolative dump, keep at most MAX_STRUCTURES_PER_DUMP of its structures, and grade them on one thread."""
    with open(dump) as dump_file:
        structures = ase.io.lammpsrun.read_lammps_dump_text(dump_file, index=slice(None), specorder=species)

    if len(structures) > MAX_STRUCTURES_PER_DUMP:
        kept = numpy.random.choice(len(structures), size=MAX_STRUCTURES_PER_DUMP, replace=False)
        structures = [structures[i] for i in kept]

    for atoms in structures:
        atoms.arrays["type_index"] = (atoms.arrays["type"] - 1).astype(numpy.int32)

    calculate_grade(potential, structures)
    return structures


def grade_extrapolative_dumps(potential, extrapolative_dumps, species=None):
    """Grade every extrapolative dump on a process pool, one dump per task and one worker per core.

    Returns the graded structures by dump, in the order the dumps finished. Workers never log.
    """
    n_workers = min(os.process_cpu_count(), len(extrapolative_dumps))
    logger.info(f"Grading {len(extrapolative_dumps)} dumps on {n_workers} workers")
    with concurrent.futures.ProcessPoolExecutor(n_workers) as pool:
        futures = {pool.submit(grade_dump, dump, potential, species): dump for dump in extrapolative_dumps}
        graded = {futures[future]: future.result() for future in concurrent.futures.as_completed(futures)}
    return graded


def max_force(atoms):
    return float(numpy.max(numpy.abs(numpy.array(atoms.calc.results["forces"]))))


def load_structures(set_name, species=None):
    with open(set_name, mode="r") as set_file:
        cfgs = read_cfg(set_file, species)
    return cfgs


def save_structures(set_name, cfgs, append=False):
    with open(set_name, mode="a" if append else "w") as set_file:
        write_cfg(set_file, cfgs)


@dataclass
class OTFCycle:
    """One OTF-MTP update cycle, from extrapolative dumps to a retrained model, stepped through by its caller.

    Inside `with`, the caller brings the active set up to date with `active_set_update`, then grades each
    dump with grade_dump against it, passes the result through preselect, selects with `selection`, evaluates
    the selected with evaluate, trains with `training`, and calls replace_potential. `active_set_update`,
    `selection(candidates)` and `training` are the keyword arguments of otf_pool's update_active_set,
    select_add and train_potential. Entering logs the otf_engine package into cycle_dir's LOG_FILE, and only
    there, and loads the OTF state and the training set; leaving records the cycle in the state's history and
    archives the cycle in cycle_dir, as failed, and re-raising, if the block raised.

    extrapolative_dumps and the options up to force_threshold are `python -m otf_engine`'s, under the same names.
    evaluator_fn : evaluator_fn(structure) labels one structure in-process; without it, or under SlurmLauncher,
                   ./evaluator.py runs on each structure
    cycle_dir    : by default the next one under otf_cycles/
    """

    extrapolative_dumps: list
    launcher: Launcher
    _: KW_ONLY
    potential: str = "potential.almtp"
    training_set: str = "train.cfg"
    species: list | None = None
    preselection_filtering: bool = True
    gamma_tolerance: float = 1.01
    gamma_max: float = 0.0
    gamma_max_cap: float = 10000.0
    extreme_lock_after_ntimes: int = 5
    max_structures: int = -1
    iteration_limit: int = 300
    force_threshold: float | None = None
    evaluator_fn: Callable | None = None
    cycle_dir: Path | None = None

    def __enter__(self):
        self.cycle_dir = self.cycle_dir or next_cycle_dir()
        package_logger = logging.getLogger(__package__)
        self._logging = package_logger.level, package_logger.propagate
        self._handler = logging.FileHandler(self.cycle_dir / LOG_FILE)
        self._handler.setFormatter(logging.Formatter(LOG_FORMAT))
        package_logger.addHandler(self._handler)
        package_logger.setLevel(logging.INFO)
        package_logger.propagate = False

        # What preselect and evaluate find, for this cycle's entry in the state's history
        self.record = dict(selection_branch="none", n_preselected=0, n_selected=0, n_evaluated=0, n_deferred=0, n_failed=0, active_set_size=None, gammas_candidates=[], gammas_selected=[], gammas_evaluated=[], max_forces_evaluated=[])
        try:
            self.state = _load_state()
            # The launcher's callbacks then only replace these keys, never add one while another thread saves.
            self.state.setdefault("timing", {})
            self.state.setdefault("memory", {})
            self.launcher.configure_timing(self.state, _save_state)
            self.launcher.configure_memory(self.state, _save_state)
            self.train_structures = load_structures(self.training_set, self.species)
        except BaseException as e:
            self.__exit__(type(e), e, e.__traceback__)
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        package_logger = logging.getLogger(__package__)
        try:
            # An interrupted cycle is neither recorded nor archived, so its dumps stay as they are.
            if exc is not None and not isinstance(exc, Exception): return False
            if isinstance(exc, JobTimedOut): logger.error("Training exhausted retries and timed out.")
            if isinstance(exc, JobOutOfMemory): logger.error("Training exhausted retries and ran out of memory.")
            # A cycle that reached its training is recorded, whether training succeeded or ran out of retries.
            if exc is None or isinstance(exc, (JobTimedOut, JobOutOfMemory)):
                self._record_history(exc)
                _save_state(self.state)
            if exc is not None: logger.error(f"Error during execution: {exc}", exc_info=(exc_type, exc, tb))
            archive_cycle(self.cycle_dir, self.potential, self.training_set, self.extrapolative_dumps, ok=exc is None)
        finally:
            package_logger.removeHandler(self._handler)
            self._handler.close()
            package_logger.setLevel(self._logging[0])
            package_logger.propagate = self._logging[1]
        return False

    def _record_history(self, exc):
        """Append this cycle's entry to the state's history and add its counts to the totals."""
        state, record = self.state, self.record
        state["n_selected_total"] = state.get("n_selected_total", 0) + record["n_selected"]
        state["n_evaluated_total"] = state.get("n_evaluated_total", 0) + record["n_evaluated"]
        state.setdefault("history", []).append({
            "cycle": int(self.cycle_dir.name.removeprefix(CYCLE_PREFIX)),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "selection_branch": record["selection_branch"],
            "n_preselected": record["n_preselected"],
            "n_selected": record["n_selected"],
            "n_evaluated": record["n_evaluated"],
            "n_deferred": record["n_deferred"],
            "n_failed": record["n_failed"],
            "n_selected_total": state["n_selected_total"],
            "n_evaluated_total": state["n_evaluated_total"],
            "training_set_size": len(self.train_structures) + record["n_evaluated"],
            "active_set_size": record["active_set_size"],
            "gammas_candidates": record["gammas_candidates"],
            "gammas_selected": record["gammas_selected"],
            "gammas_evaluated": record["gammas_evaluated"],
            "max_forces_evaluated": record["max_forces_evaluated"],
            "gamma_max0": state.get("gamma_max0"),
            "training_timed_out": isinstance(exc, JobTimedOut),
            "training_out_of_memory": isinstance(exc, JobOutOfMemory),
            **self.launcher.timing.cycle_times(),
        })

    def preselect(self, graded):
        """Reduce the graded structures to the candidates for selection: the gamma policy, then a random cap.

        graded : the structures grade_dump returned, by dump
        """
        state = self.state
        # The size of the active set the dumps were graded against, recorded with the cycle
        self.record["active_set_size"] = len(read_mvs_state(self.potential).selected_cfgs)
        candidates = []
        for k, (dump, structures) in enumerate(graded.items(), 1):
            logger.info(f"Graded dump {k}/{len(graded)}: {dump} with {len(structures)} structures")
            candidates += structures

        gammas = numpy.array([atoms.info["features"]["MV_grade"] for atoms in candidates])
        self.record["gammas_candidates"] = gammas.tolist()
        branch = "none"

        # The gamma policy: of the candidates above gamma_tolerance, those below gamma_max ("normal"), or else
        # the lowest, when it lies below gamma_max0 ("intermediate") or extremes are still allowed ("extreme").
        if self.preselection_filtering:
            above = gammas > self.gamma_tolerance
            candidates, gammas = [atoms for atoms, a in zip(candidates, above) if a], gammas[above]
            logger.info(f"Preselection: {len(candidates)}/{len(above)} structures above gamma_tolerance={self.gamma_tolerance:.4f}")

            if candidates:
                gamma_max0 = state.get("gamma_max0", self.gamma_max_cap)
                lowest = int(numpy.argmin(gammas))
                min_gamma = gammas[lowest]
                if numpy.any(gammas < self.gamma_max):
                    candidates = [atoms for atoms, g in zip(candidates, gammas) if g < self.gamma_max]
                    branch = "normal"
                elif min_gamma < gamma_max0:
                    logger.info(f"gamma_max0 = {gamma_max0:.4f} (history length = {len(state.get('gamma_max0_history', []))})")
                    candidates = [candidates[lowest]]
                    branch = "intermediate"
                    logger.info(f"Selected structure with gamma = {min_gamma:.4f}")
                else:
                    extreme_allowed = state.get("extreme_allowed", True)
                    consecutive_non_extreme = state.get("consecutive_non_extreme", 0)
                    state["extreme_count"] = state.get("extreme_count", 0) + 1
                    logger.warning(f"Extreme Warning: all gammas > gamma_max0={gamma_max0:.4f}, min gamma = {min_gamma:.4f}, consecutive_non_extreme={consecutive_non_extreme} (lock_after={self.extreme_lock_after_ntimes}), extreme_allowed={extreme_allowed}")
                    candidates = [candidates[lowest]] if extreme_allowed else []
                    if extreme_allowed:
                        state["consecutive_non_extreme"] = 0
                        branch = "extreme"
                        logger.info(f"Selecting structure with gamma = {min_gamma:.4f}")
                    else:
                        logger.warning(f"Skipping selection: {consecutive_non_extreme} consecutive non-extreme iterations reached limit of {self.extreme_lock_after_ntimes}")

                # After extreme_lock_after_ntimes cycles in a row without an extreme one, extremes are never selected again.
                if branch in ("normal", "intermediate"):
                    state["consecutive_non_extreme"] = state.get("consecutive_non_extreme", 0) + 1
                    if state["consecutive_non_extreme"] >= self.extreme_lock_after_ntimes: state["extreme_allowed"] = False

                # gamma_max0 is the mean of the last 10 lowest gammas recorded here, and never below gamma_max.
                if numpy.all(gammas > self.gamma_max) and min_gamma < self.gamma_max_cap:
                    history = (state.get("gamma_max0_history", []) + [float(min_gamma)])[-10:]
                    state["gamma_max0_history"] = history
                    state["gamma_max0_full_history"] = state.get("gamma_max0_full_history", []) + [float(min_gamma)]
                    state["gamma_max0"] = max(numpy.mean(history), self.gamma_max)
                    logger.info(f"Updated gamma_max0: {gamma_max0:.4f} -> {state['gamma_max0']:.4f}")

                logger.info(f"Post-preselection: {len(candidates)} structures selected")

        # A random cap on the total
        if 0 < self.max_structures < len(candidates):
            kept = numpy.random.choice(len(candidates), size=self.max_structures, replace=False)
            candidates = [candidates[i] for i in kept]
            logger.info(f"Post-preselection max-structures: {len(candidates)}")

        self.record["selection_branch"] = branch
        self.record["n_preselected"] = len(candidates)
        return candidates

    @property
    def active_set_update(self):
        """The active-set update the dumps must be graded after, as otf_pool.update_active_set's keyword arguments."""
        return dict(potential=self.potential, training_set=self.training_set, species=self.species)

    def selection(self, candidates):
        """The selection from the candidates of those that extend the active set, as otf_pool.select_add's keyword arguments."""
        return dict(potential=self.potential, training_set=self.training_set, candidates=candidates, species=self.species)

    def evaluate(self, selected):
        """Evaluate the selected structures and those the previous cycle deferred, add those that succeed to the training set, log the potential's errors on them, and return their count."""
        deferred_file = self.cycle_dir.parent / DEFERRED_EVALS_FILE
        resubmitted = ase.io.read(deferred_file, index=":", format="extxyz") if deferred_file.is_file() else []
        structures = selected + resubmitted
        n = len(structures)
        w = len(str(n))
        parallel = self.launcher.concurrent_eval and n > 1
        logger.info(f"Evaluating {n} structures {'concurrently' if parallel else 'sequentially'}.")
        evaluated, deferred, gammas_evaluated = [],[],[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=None if parallel else 1) as executor:
            futures = {executor.submit(self._evaluate_one, i, s): i for i, s in enumerate(structures)}
            for k, future in enumerate(concurrent.futures.as_completed(futures), 1):
                i = futures[future]
                status, result = future.result()
                logger.info(f"[{k:{w}d}/{n}] struct {i+1:{w}d} — {status}")
                if status == "deferred":
                    deferred += [structures[i]]
                elif status == "ok":
                    evaluated += [result]
                    save_structures(self.training_set, [result], append=True)
                    gammas_evaluated += [structures[i].info["features"]["MV_grade"]]
        n_failed = n - len(evaluated) - len(deferred)
        logger.info(f"Evaluated {len(evaluated)}/{n} successfully ({len(deferred)} deferred, {n_failed} failed).")
        self.record.update(n_selected=n, n_evaluated=len(evaluated), n_deferred=len(deferred), n_failed=n_failed, gammas_selected=[s.info["features"]["MV_grade"] for s in structures], gammas_evaluated=gammas_evaluated, max_forces_evaluated=[max_force(atoms) for atoms in evaluated])

        ase.io.write(deferred_file, deferred, format="extxyz")
        if evaluated: logger.info(errors(self.potential, evaluated, "Errors on the evaluated structures:", first=len(self.train_structures) + 1, species=self.species))
        else: logger.info("No configurations selected or evaluated — retraining.")
        return len(evaluated)

    def _evaluate_one(self, i, structure):
        """Evaluate one structure: ("ok", result), ("skipped", None) above force_threshold, ("failed", None), or ("deferred", None) when it timed out or ran out of memory at most launcher.max_retries times."""
        eval_dir = self.cycle_dir / f"eval_{i:03d}"
        try:
            result = self.launcher.call_evaluator(self.evaluator_fn, structure, eval_dir, timed_out_at_s=structure.info.get("eval_timed_out_at_s"))
            if self.force_threshold is not None and max_force(result) > self.force_threshold:
                logger.warning(f"struct {i+1}: skipped (max force {max_force(result):.2f} eV/Å exceeds threshold)")
                return "skipped", None
            return "ok", result
        except (JobTimedOut, JobOutOfMemory) as e:
            if isinstance(e, JobTimedOut) and e.time_limit_s: structure.info["eval_timed_out_at_s"] = e.time_limit_s
            structure.info["eval_attempts"] = int(structure.info.get("eval_attempts", 0)) + 1
            deferred = structure.info["eval_attempts"] <= self.launcher.max_retries
            logger.warning(f"struct {i+1}: {'timed out' if isinstance(e, JobTimedOut) else 'ran out of memory'} on attempt {structure.info['eval_attempts']}, {'deferred to the next cycle' if deferred else 'dropped'}")
            return ("deferred" if deferred else "failed"), None
        except Exception as e:
            eval_dir.mkdir(parents=True, exist_ok=True)
            with open(eval_dir / "eval.log", "a") as _f:
                traceback.print_exc(file=_f)
            try:
                logger.error(f"struct {i+1} espresso.err:\n{(eval_dir / 'espresso.err').read_text()}")
                shutil.rmtree(eval_dir / "pwscf.save")
            except Exception:
                pass
            logger.error(f"struct {i+1}: failed: {e}")
            return "failed", None

    @property
    def training(self):
        """The training this cycle needs, as otf_pool.train_potential's keyword arguments."""
        return dict(potential=self.potential, training_set=self.training_set, save_to=f"tmp_{self.potential}", species=self.species, settings={"iteration_limit": self.iteration_limit, "log": TRAIN_LOG, "species": ",".join(self.species or [])})

    def replace_potential(self):
        """Replace the potential with the one training wrote."""
        os.replace(self.training["save_to"], self.potential)
        logger.info(f"OTF-MTP update cycle complete. New potential saved to {self.potential}.")


def run_cycle(extrapolative_dumps, launcher: Launcher, *, mlp_command, **options):
    """Run one OTFCycle as `python -m otf_engine` does: grading on a process pool, training with mlp_command's `mlp train` through the launcher.

    options : OTFCycle's
    """
    with OTFCycle(extrapolative_dumps, launcher, **options) as cycle:
        train_eqns = update_active_set(cycle.potential, cycle.train_structures)
        candidates = cycle.preselect(grade_extrapolative_dumps(cycle.potential, extrapolative_dumps, cycle.species))
        selected, _ = select_add(cycle.potential, cycle.train_structures, candidates, train_eqns=train_eqns)
        n_ok = cycle.evaluate(selected)
        launcher.run(f"{mlp_command} train {cycle.potential} {cycle.training_set} --save_to={cycle.training['save_to']} --iteration_limit={cycle.iteration_limit}", log_file=TRAIN_LOG, training_set_size=len(cycle.train_structures) + n_ok)
        cycle.replace_potential()
