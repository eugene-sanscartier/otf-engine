import concurrent.futures
import datetime
import json
import logging
import os
import shutil
import traceback
from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass
from pathlib import Path

import numpy

import ase
import ase.io.lammpsrun

from .io_cfg import read_cfg, write_cfg
from .mtp_backend import calculate_grade, select_add, update_active_set
from .almtp_io import read_mvs_state
from .cycles import LOG_FILE, LOG_FORMAT, archive_cycle, current_cycle_dir, next_cycle_dir
from .launchers import Launcher, JobTimedOut, JobOutOfMemory

logger = logging.getLogger(__name__)
_EVAL_TIMED_OUT = object()

OTF_STATE_FILE = "otf_state.json"


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
    with open(OTF_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


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


def _record_state(state, n_train, active_set_size):
    timing = state.get("timing", {})
    cycle_dir = current_cycle_dir()
    cycle = int(cycle_dir.name.split("_")[-1]) if cycle_dir is not None else len(state.get("history", []))
    n_selected = state.pop("n_selected", 0)
    n_ok = state.pop("n_ok", 0)
    state["n_selected_total"] = state.get("n_selected_total", 0) + n_selected
    state["n_evaluated_total"] = state.get("n_evaluated_total", 0) + n_ok
    state.setdefault("history", []).append({
        "cycle": cycle,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "selection_branch": state.pop("selection_branch", "none"),
        "n_preselected": state.pop("n_preselected", 0),
        "n_selected": n_selected,
        "n_evaluated": n_ok,
        "n_timed_out": state.pop("n_timed_out", 0),
        "n_failed": state.pop("n_failed", 0),
        "n_selected_total": state["n_selected_total"],
        "n_evaluated_total": state["n_evaluated_total"],
        "training_set_size": n_train + n_ok,
        "active_set_size": active_set_size,
        "gammas_candidates": state.pop("gammas_candidates", []),
        "gammas_selected": state.pop("gammas_selected", []),
        "gammas_evaluated": state.pop("gammas_evaluated", []),
        "max_forces_evaluated": state.pop("max_forces_evaluated", []),
        "gamma_max0": state.get("gamma_max0"),
        "training_timed_out": state.pop("training_timed_out", False),
        "training_out_of_memory": state.pop("training_out_of_memory", False),
        "eval_time_s": timing.get("eval_time_s"),
        "eval_time_alloc_s": timing.get("eval_time_alloc_s"),
        "train_time_s": timing.get("train_time_s"),
        "train_time_alloc_s": timing.get("train_time_alloc_s"),
    })


def max_force(atoms):
    return float(numpy.max(numpy.abs(numpy.array(atoms.calc.results["forces"]))))


def forcesthr_excess(atoms, threshold):
    if atoms.calc is None or "forces" not in atoms.calc.results:
        return False
    return max_force(atoms) > threshold


def load_structures(set_name, species=None):
    with open(set_name, mode="r") as set_file:
        cfgs = read_cfg(set_file, species)
    return cfgs


def save_structures(set_name, cfgs, append=False):
    with open(set_name, mode="a" if append else "w") as set_file:
        write_cfg(set_file, cfgs)


def _eval_one(i, structure, evaluator_fn, launcher, force_threshold):
    cycle = current_cycle_dir()
    eval_dir = (cycle / f"eval_{i:03d}") if cycle is not None else Path(f"eval_{i:03d}")
    try:
        result = launcher.call_evaluator(evaluator_fn, structure, eval_dir)
        if force_threshold is not None and forcesthr_excess(result, threshold=force_threshold):
            logger.warning(f"struct {i+1}: skipped (max force {max_force(result):.2f} eV/Å exceeds threshold)")
            return None
        return result
    except JobTimedOut:
        logger.warning(f"struct {i+1}: timed out")
        return _EVAL_TIMED_OUT
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
        return None


def eval_structures(selected_structures, training_set, evaluator_fn, launcher, force_threshold=None, state=None):
    n = len(selected_structures)
    w = len(str(n)) if n else 1
    parallel = launcher.concurrent_eval and n > 1
    logger.info(f"Evaluating {n} structures {'concurrently' if parallel else 'sequentially'}.")
    n_ok = n_timed_out = 0
    gammas_evaluated = []
    max_forces_evaluated = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=None if parallel else 1) as executor:
        futures = {executor.submit(_eval_one, i, s, evaluator_fn, launcher, force_threshold): i for i, s in enumerate(selected_structures)}
        for k, future in enumerate(concurrent.futures.as_completed(futures), 1):
            i = futures[future]
            result = future.result()
            logger.info(f"[{k:{w}d}/{n}] struct {i+1:{w}d} — {'ok' if result not in (None, _EVAL_TIMED_OUT) else 'timed out' if result is _EVAL_TIMED_OUT else 'failed'}")
            if result is _EVAL_TIMED_OUT:
                n_timed_out += 1
            elif result is not None:
                n_ok += 1
                save_structures(training_set, [result], append=True)
                gammas_evaluated += [selected_structures[i].info["features"]["MV_grade"]]
                max_forces_evaluated += [max_force(result)]
    logger.info(f"Evaluated {n_ok}/{n} successfully ({n_timed_out} timed out, {n - n_ok - n_timed_out} failed).")
    if state is not None:
        state["n_selected"] = n
        state["n_ok"] = n_ok
        state["n_timed_out"] = n_timed_out
        state["n_failed"] = n - n_ok - n_timed_out
        state["gammas_selected"] = [s.info["features"]["MV_grade"] for s in selected_structures]
        state["gammas_evaluated"] = gammas_evaluated
        state["max_forces_evaluated"] = max_forces_evaluated
    return n_ok


@dataclass
class OTFCycle:
    """One OTF-MTP update cycle, from extrapolative dumps to a retrained model, stepped through by its caller.

    Inside `with`, the caller grades each dump with grade_dump, passes the result through preselect, select
    and evaluate, trains with `training`, and calls replace_potential. Entering logs the otf_engine package
    into cycle_dir's LOG_FILE, and only there, loads the OTF state and brings the active set up to date;
    leaving records the state and archives the cycle in cycle_dir, as failed, and re-raising, if the block raised.

    extrapolative_dumps and the options up to force_threshold are `python -m otf_engine`'s, under the same names.
    evaluator_fn : evaluator_fn(structure) labels one structure, for the launchers that call it in-process;
                   SlurmLauncher runs ./evaluator.py as a job of its own instead
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

        try:
            self.state = _load_state()
            self.launcher.configure_timing(self.state, _save_state)
            self.launcher.configure_memory(self.state, _save_state)
            # The dumps are graded against the active set, so it is brought up to date with the training set first.
            self.train_structures = load_structures(self.training_set, self.species)
            self.train_eqns = update_active_set(self.potential, self.train_structures)
            self.active_set_size = len(read_mvs_state(self.potential).selected_cfgs)
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
                self.state["timing"] = self.launcher.timing.to_dict()
                self.state["training_timed_out"] = isinstance(exc, JobTimedOut)
                self.state["training_out_of_memory"] = isinstance(exc, JobOutOfMemory)
                _record_state(self.state, len(self.train_structures), self.active_set_size)
                _save_state(self.state)
            if exc is not None: logger.error(f"Error during execution: {exc}", exc_info=(exc_type, exc, tb))
            archive_cycle(self.cycle_dir, self.potential, self.training_set, self.extrapolative_dumps, ok=exc is None)
        finally:
            package_logger.removeHandler(self._handler)
            self._handler.close()
            package_logger.setLevel(self._logging[0])
            package_logger.propagate = self._logging[1]
        return False

    def preselect(self, graded):
        """Reduce the graded structures to the candidates for selection: the gamma policy, then a random cap.

        graded : the structures grade_dump returned, by dump
        """
        state = self.state
        candidates = []
        for k, (dump, structures) in enumerate(graded.items(), 1):
            logger.info(f"Graded dump {k}/{len(graded)}: {dump} with {len(structures)} structures")
            candidates += structures

        gammas = numpy.array([atoms.info["features"]["MV_grade"] for atoms in candidates])
        state["gammas_candidates"] = gammas.tolist()
        state["selection_branch"] = "none"

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
                    state["selection_branch"] = "normal"
                elif min_gamma < gamma_max0:
                    logger.info(f"gamma_max0 = {gamma_max0:.4f} (history length = {len(state.get('gamma_max0_history', []))})")
                    candidates = [candidates[lowest]]
                    state["selection_branch"] = "intermediate"
                    logger.info(f"Selected structure with gamma = {min_gamma:.4f}")
                else:
                    extreme_allowed = state.get("extreme_allowed", True)
                    consecutive_non_extreme = state.get("consecutive_non_extreme", 0)
                    state["extreme_count"] = state.get("extreme_count", 0) + 1
                    logger.warning(f"Extreme Warning: all gammas > gamma_max0={gamma_max0:.4f}, min gamma = {min_gamma:.4f}, consecutive_non_extreme={consecutive_non_extreme} (lock_after={self.extreme_lock_after_ntimes}), extreme_allowed={extreme_allowed}")
                    candidates = [candidates[lowest]] if extreme_allowed else []
                    if extreme_allowed:
                        state["consecutive_non_extreme"] = 0
                        state["selection_branch"] = "extreme"
                        logger.info(f"Selecting structure with gamma = {min_gamma:.4f}")
                    else:
                        logger.warning(f"Skipping selection: {consecutive_non_extreme} consecutive non-extreme iterations reached limit of {self.extreme_lock_after_ntimes}")

                # After extreme_lock_after_ntimes cycles in a row without an extreme one, extremes are never selected again.
                if state["selection_branch"] in ("normal", "intermediate"):
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

        state["n_preselected"] = len(candidates)
        return candidates

    def select(self, candidates):
        """Select, from the candidates, those that extend the active set, as mlip-3's select_add does."""
        # train_eqns were built on entry, from the same coefficients and weights.
        selected, _ = select_add(self.potential, self.train_structures, candidates, train_eqns=self.train_eqns)
        return selected

    def evaluate(self, selected):
        """Evaluate the selected structures through the launcher, add those that succeed to the training set, and return their count."""
        n_ok = eval_structures(selected, self.training_set, self.evaluator_fn, self.launcher, force_threshold=self.force_threshold, state=self.state)
        if not n_ok: logger.info("No configurations selected or evaluated — retraining.")
        return n_ok

    @property
    def training(self):
        """The training this cycle needs, as otf_pool.train_potential's keyword arguments."""
        return dict(potential=self.potential, training_set=self.training_set, save_to=f"tmp_{self.potential}", species=self.species, settings={"iteration_limit": self.iteration_limit, "log": "mlip_train.log"})

    def replace_potential(self):
        """Replace the potential with the one training wrote."""
        os.replace(self.training["save_to"], self.potential)
        logger.info(f"OTF-MTP update cycle complete. New potential saved to {self.potential}.")


def run_cycle(extrapolative_dumps, launcher: Launcher, *, mlp_command=None, **options):
    """Run one OTFCycle as `python -m otf_engine` does: grading on a process pool, training with mlp_command's `mlp train` through the launcher.

    options : OTFCycle's
    """
    with OTFCycle(extrapolative_dumps, launcher, **options) as cycle:
        candidates = cycle.preselect(grade_extrapolative_dumps(cycle.potential, extrapolative_dumps, cycle.species))
        n_ok = cycle.evaluate(cycle.select(candidates))
        launcher.run(f"{mlp_command} train {cycle.potential} {cycle.training_set} --save_to={cycle.training['save_to']} --iteration_limit={cycle.iteration_limit} ", log_file="mlip_train.log", training_set_size=len(cycle.train_structures) + n_ok)
        cycle.replace_potential()
