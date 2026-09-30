import concurrent.futures
import contextlib
import datetime
import functools
import json
import logging
import os
import shutil
import traceback
from pathlib import Path

import numpy

import ase
import ase.io.lammpsrun

from .io_cfg import read_cfg, write_cfg
from .mtp_backend import calculate_grade, select_add, update_active_set
from .almtp_io import read_mvs_state
from .cycles import current_cycle_dir, next_cycle_dir, recorded_in
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
    """Parse one extrapolative dump, keep at most MAX_STRUCTURES_PER_DUMP of its structures, and grade them."""
    with open(dump) as dump_file:
        structures = ase.io.lammpsrun.read_lammps_dump_text(dump_file, index=slice(None), specorder=species)

    if len(structures) > MAX_STRUCTURES_PER_DUMP:
        kept = numpy.random.choice(len(structures), size=MAX_STRUCTURES_PER_DUMP, replace=False)
        structures = [structures[i] for i in kept]

    for atoms in structures:
        atoms.arrays["type_index"] = (atoms.arrays["type"] - 1).astype(numpy.int32)

    calculate_grade(potential, structures)
    return structures


def grade_extrapolative_dumps(potential, extrapolative_dumps, species=None, submit_grade=None):
    """Grade every extrapolative dump, one task per dump, and return the graded structures.

    submit_grade : submit_grade(dump=, potential=, species=) runs grade_dump as a task and returns its
                   Future; by default a process pool runs them, one worker per core. Tasks never log.
    """
    pool = contextlib.nullcontext()
    if submit_grade is None:
        # Pin each worker's BLAS to one thread, or they oversubscribe the cores the pool already claims.
        # This has to happen before the pool exists: a worker imports numpy while resolving the task
        # function, and BLAS fixes its thread count then.
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"

        n_workers = min(os.process_cpu_count(), len(extrapolative_dumps))
        logger.info(f"Grading {len(extrapolative_dumps)} dumps on {n_workers} workers")
        pool = concurrent.futures.ProcessPoolExecutor(n_workers)
        submit_grade = functools.partial(pool.submit, grade_dump)

    graded_structures = []
    with pool:
        futures = {submit_grade(dump=dump, potential=potential, species=species): dump for dump in extrapolative_dumps}
        for k, future in enumerate(concurrent.futures.as_completed(futures), 1):
            structures = future.result()
            logger.info(f"Graded dump {k}/{len(futures)}: {futures[future]} with {len(structures)} structures")
            graded_structures += structures

    return graded_structures


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


def _record_non_extreme(state, extreme_lock_after_ntimes):
    state["consecutive_non_extreme"] = state.get("consecutive_non_extreme", 0) + 1
    if state["consecutive_non_extreme"] >= extreme_lock_after_ntimes:
        state["extreme_allowed"] = False


def preselected_filter(cfgs, gamma_tolerance, gamma_max, gamma_max_cap, state, extreme_lock_after_ntimes=10):
    gamma_max0 = state.get("gamma_max0", gamma_max_cap)
    n_total = len(cfgs)

    gammas = numpy.array([cfg.info["features"]["MV_grade"] for cfg in cfgs])
    mask = gammas > gamma_tolerance
    cfgs = [cfg for cfg, m in zip(cfgs, mask) if m]
    gammas = gammas[mask]
    logger.info(f"Preselection: {len(cfgs)}/{n_total} structures above gamma_tolerance={gamma_tolerance:.4f}")

    if not cfgs:
        state["selection_branch"] = "none"
        return []

    min_gamma = numpy.min(gammas)
    filtred_cfgs = []
    state["selection_branch"] = "none"

    if numpy.any(gammas < gamma_max):
        filtred_cfgs = [cfg for cfg, g in zip(cfgs, gammas) if g < gamma_max]
        state["selection_branch"] = "normal"
        _record_non_extreme(state, extreme_lock_after_ntimes)

    elif numpy.any(gammas < gamma_max0):
        logger.info(f"gamma_max0 = {gamma_max0:.4f} (history length = {len(state.get('gamma_max0_history', []))})")
        idx = numpy.argmin(gammas)
        filtred_cfgs = [cfgs[idx]]
        state["selection_branch"] = "intermediate"
        logger.info(f"Selected structure with gamma = {gammas[idx]:.4f}")
        _record_non_extreme(state, extreme_lock_after_ntimes)

    else:
        extreme_allowed = state.get("extreme_allowed", True)
        consecutive_non_extreme = state.get("consecutive_non_extreme", 0)
        state["extreme_count"] = state.get("extreme_count", 0) + 1
        logger.warning(f"Extreme Warning: all gammas > gamma_max0={gamma_max0:.4f}, min gamma = {min_gamma:.4f}, consecutive_non_extreme={consecutive_non_extreme} (lock_after={extreme_lock_after_ntimes}), extreme_allowed={extreme_allowed}")
        if extreme_allowed:
            filtred_cfgs = [cfgs[numpy.argmin(gammas)]]
            state["consecutive_non_extreme"] = 0
            state["selection_branch"] = "extreme"
            logger.info(f"Selecting structure with gamma = {min_gamma:.4f}")
        else:
            logger.warning(f"Skipping selection: {consecutive_non_extreme} consecutive non-extreme iterations reached limit of {extreme_lock_after_ntimes}")

    # gamma_max0 is the mean of the last 10 lowest gammas recorded here, and never below gamma_max.
    if numpy.all(gammas > gamma_max) and min_gamma < gamma_max_cap:
        history = (state.get("gamma_max0_history", []) + [float(min_gamma)])[-10:]
        state["gamma_max0_history"] = history
        state["gamma_max0_full_history"] = state.get("gamma_max0_full_history", []) + [float(min_gamma)]
        state["gamma_max0"] = max(numpy.mean(history), gamma_max)
        logger.info(f"Updated gamma_max0: {gamma_max0:.4f} -> {state['gamma_max0']:.4f}")

    logger.info(f"Post-preselection: {len(filtred_cfgs)} structures selected")

    return filtred_cfgs


def max_structureselection(filtred_cfgs, max_structures=-1):
    if max_structures > 0 and len(filtred_cfgs) > max_structures:
        rnd_selected = numpy.random.choice(len(filtred_cfgs), size=max_structures, replace=False)
        filtred_cfgs = [filtred_cfgs[i] for i in rnd_selected]
        logger.info(f"Post-preselection max-structures: {len(filtred_cfgs)}")
    return filtred_cfgs


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


def run_cycle(extrapolative_dumps, launcher: Launcher, *, potential="potential.almtp", training_set="train.cfg", species=None, preselection_filtering=True, gamma_tolerance=1.01, gamma_max=0.0, gamma_max_cap=10000.0, extreme_lock_after_ntimes=5, max_structures=-1, iteration_limit=300, force_threshold=None, evaluator_fn=None, mlp_command=None, submit_grade=None, train_potential=None, cycle_dir=None):
    """Run one OTF-MTP update cycle from extrapolative dumps to a retrained model, logged and archived in cycle_dir.

    A failed cycle is archived as failed, then raised. extrapolative_dumps and the options up to
    force_threshold are `python -m otf_engine`'s, under the same names.

    evaluator_fn    : evaluator_fn(structure) labels one structure, for the launchers that call it
                      in-process; SlurmLauncher runs ./evaluator.py as a job of its own instead
    mlp_command     : the `mlp` that trains, through the launcher, when train_potential is not given
    submit_grade    : submit_grade(dump=, potential=, species=) runs grade_dump as a task and returns its
                      Future; by default a process pool grades the dumps
    train_potential : train_potential(potential=, training_set=, save_to=, species=, settings=) trains the
                      potential
    cycle_dir       : by default the next one under otf_cycles/
    """
    with recorded_in(cycle_dir or next_cycle_dir(), potential, training_set, extrapolative_dumps):
        state = _load_state()
        launcher.configure_timing(state, _save_state)
        launcher.configure_memory(state, _save_state)

        # Step 1: ensure the active set is consistent with the current training set.
        train_structures = load_structures(training_set, species)
        train_eqns = update_active_set(potential, train_structures)
        active_set_size = len(read_mvs_state(potential).selected_cfgs)

        # Step 2: parse the extrapolative dumps and grade them against that active set.
        candidate_structures = grade_extrapolative_dumps(potential, extrapolative_dumps, species=species, submit_grade=submit_grade)

        # Step 3: optionally apply preselection policy.
        state["selection_branch"] = "none"
        state["gammas_candidates"] = [c.info["features"]["MV_grade"] for c in candidate_structures]
        if preselection_filtering:
            candidate_structures = preselected_filter(candidate_structures, gamma_tolerance, gamma_max, gamma_max_cap, extreme_lock_after_ntimes=extreme_lock_after_ntimes, state=state)

        # Step 4: optionally cap the surviving pool size.
        if max_structures > 0:
            candidate_structures = max_structureselection(candidate_structures, max_structures=max_structures)

        # Step 5: run the structure-selection step.
        # train_eqns were built in step 1 from the same coefficients and weights.
        selected_structures, _ = select_add(potential, train_structures, candidate_structures, train_eqns=train_eqns)

        # Step 6: evaluate the selected structures.
        n_ok = eval_structures(selected_structures, training_set, evaluator_fn, launcher, force_threshold=force_threshold, state=state)
        if not n_ok:
            logger.info("No configurations selected or evaluated — retraining.")

        # Step 7: retrain the potential on the updated training set.
        train_exc = None
        try:
            if train_potential is None:
                launcher.run(f"{mlp_command} train {potential} {training_set} --save_to=tmp_{potential} --iteration_limit={iteration_limit} ", log_file="mlip_train.log", training_set_size=len(train_structures) + n_ok)
            else:
                train_potential(potential=potential, training_set=training_set, save_to=f"tmp_{potential}", species=species, settings={"iteration_limit": iteration_limit, "log": "mlip_train.log"})
        except JobTimedOut as exc:
            train_exc = exc
            logger.error("Training exhausted retries and timed out.")
        except JobOutOfMemory as exc:
            train_exc = exc
            logger.error("Training exhausted retries and ran out of memory.")
        else:
            os.replace(f"tmp_{potential}", potential)
            logger.info(f"OTF-MTP update cycle complete. New potential saved to {potential}.")

        state["timing"] = launcher.timing.to_dict()
        state["n_preselected"] = len(candidate_structures)
        state["training_timed_out"] = isinstance(train_exc, JobTimedOut)
        state["training_out_of_memory"] = isinstance(train_exc, JobOutOfMemory)
        _record_state(state, len(train_structures), active_set_size)
        _save_state(state)

        if train_exc is not None:
            raise train_exc
