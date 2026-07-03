import concurrent.futures
import datetime
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
from .cycles import current_cycle_dir
from .launchers import Launcher, JobTimedOut

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


def load_extrapolative_dumps(extrapolative_dumps, extrapolation_field="f_extrapolation_grade", species=None):
    collected_dumps = []
    for extrapolative_dump in extrapolative_dumps:
        with open(extrapolative_dump) as dump_file:
            dumps = ase.io.lammpsrun.read_lammps_dump_text(dump_file, index=slice(None), specorder=species)
            logger.info(f"Reading extrapolative dump: {extrapolative_dump} with {len(dumps)} structures")

            if len(dumps) > 100:
                logger.warning(f"Large extrapolative dump with {len(dumps)} structures, this may cause performance issues.")
                _indices = numpy.random.choice(len(dumps), size=100, replace=False)
                dumps = [dumps[i] for i in _indices]

            collected_dumps += dumps

    for dump in collected_dumps:
        if dump.has(extrapolation_field):
            dump.set_array("nbh_grades", dump.get_array(extrapolation_field).flatten())
    return collected_dumps


def _update_gamma_max0(state, obs, gamma_max0_floor, gamma_max0_window=10):
    """Add obs to rolling history, return updated gamma_max0 (never below gamma_max0_floor)."""
    history = state.get("gamma_max0_history", [])
    history = (history + [float(obs)])[-gamma_max0_window:]
    gamma_max0_new = max(numpy.mean(history), gamma_max0_floor)
    state["gamma_max0_history"] = history
    state["gamma_max0"] = gamma_max0_new
    full = state.get("gamma_max0_full_history", [])
    full += [float(obs)]
    state["gamma_max0_full_history"] = full
    return gamma_max0_new


def _record_state(state, n_train, active_set_size, timing_stats=None):
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
        **(timing_stats or {}),
    })


def max_force(atoms):
    return float(numpy.max(numpy.abs(numpy.array(atoms.calc.results["forces"]))))


def forcesthr_excess(atoms, threshold):
    if atoms.calc is None or "forces" not in atoms.calc.results:
        return False
    return max_force(atoms) > threshold


def _record_non_extreme(state, extreme_lock_after_ntimes, _save=True):
    state["consecutive_non_extreme"] = state.get("consecutive_non_extreme", 0) + 1
    if state["consecutive_non_extreme"] >= extreme_lock_after_ntimes:
        state["extreme_allowed"] = False
    if _save:
        _save_state(state)


def _checkgrade(cfg):
    if "nbh_grades" in cfg.arrays:
        return cfg.arrays["nbh_grades"].max()
    if "features" in cfg.info and "MV_grade" in cfg.info["features"]:
        return cfg.info["features"]["MV_grade"]
    return 0


def preselected_filter(cfgs, gamma_tolerance, gamma_max, gamma_max_cap, extreme_lock_after_ntimes=10, state=None):
    _own_state = state is None
    if _own_state:
        state = _load_state()
    gamma_max0 = state.get("gamma_max0", gamma_max_cap)
    n_total = len(cfgs)

    gammas = numpy.array([_checkgrade(cfg) for cfg in cfgs])
    state["gammas_candidates"] = gammas.tolist()
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
        _record_non_extreme(state, extreme_lock_after_ntimes, _save=_own_state)

    elif numpy.any(gammas < gamma_max0):
        logger.info(f"gamma_max0 = {gamma_max0:.4f} (history length = {len(state.get('gamma_max0_history', []))})")
        idx = numpy.argmin(gammas)
        filtred_cfgs = [cfgs[idx]]
        state["selection_branch"] = "intermediate"
        logger.info(f"Selected structure with gamma = {gammas[idx]:.4f}")
        _record_non_extreme(state, extreme_lock_after_ntimes, _save=_own_state)

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
        if _own_state:
            _save_state(state)

    if numpy.all(gammas > gamma_max) and min_gamma < gamma_max_cap:
        gamma_max0_new = _update_gamma_max0(state, min_gamma, gamma_max)
        logger.info(f"Updated gamma_max0: {gamma_max0:.4f} -> {gamma_max0_new:.4f}")
        if _own_state:
            _save_state(state)

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


def main(args, launcher: Launcher = None, mlp_command=None, evaluator_fn=None):
    """Run one OTF-MTP update cycle from extrapolative dumps to a retrained model."""

    state = _load_state()
    launcher.configure_timing(state, _save_state)

    # Step 1: load the extrapolative structures emitted by the upstream run.
    candidate_structures = load_extrapolative_dumps(args.extrapolative_dumps, species=args.species)

    # Step 1b: ensure the active set is consistent with the current training set.
    train_structures = load_structures(args.training_set, args.species)
    update_active_set(args.potential, train_structures)
    active_set_size = len(read_mvs_state(args.potential).selected_cfgs)

    # Step 2: ensure every candidate carries an extrapolation grade.
    candidate_structures = calculate_grade(args.potential, candidate_structures)

    # Step 3: optionally apply preselection policy.
    state["selection_branch"] = "none"
    state["gammas_candidates"] = [c.info["features"]["MV_grade"] for c in candidate_structures]
    if args.preselection_filtering:
        candidate_structures = preselected_filter(
            candidate_structures, args.gamma_tolerance, args.gamma_max, args.gamma_max_cap,
            extreme_lock_after_ntimes=args.extreme_lock_after_ntimes, state=state)

    # Step 4: optionally cap the surviving pool size.
    if args.max_structures > 0:
        candidate_structures = max_structureselection(candidate_structures, max_structures=args.max_structures)

    # Step 5: run the structure-selection step.
    selected_structures, _ = select_add(args.potential, train_structures, candidate_structures)

    # Step 6: evaluate the selected structures.
    n_ok = eval_structures(selected_structures, args.training_set, evaluator_fn, launcher, force_threshold=args.force_threshold, state=state)
    if not n_ok:
        logger.info("No configurations selected or evaluated — retraining.")

    # Step 7: retrain the potential on the updated training set.
    train_exc = None
    try:
        launcher.run(f"{mlp_command} train {args.potential} {args.training_set} --save_to=tmp_{args.potential} --iteration_limit={args.iteration_limit} ", log_file="mlip_train.log", training_set_size=len(train_structures) + n_ok)
    except JobTimedOut as exc:
        train_exc = exc
        logger.error("Training exhausted retries and timed out — recording cycle as training_timed_out.")
    else:
        os.replace(f"tmp_{args.potential}", args.potential)
        logger.info(f"OTF-MTP update cycle complete. New potential saved to {args.potential}.")

    state["timing"] = launcher.timing.to_dict()
    state["n_preselected"] = len(candidate_structures)
    state["training_timed_out"] = train_exc is not None
    _record_state(state, len(train_structures), active_set_size, timing_stats={**launcher.timing._last_eval, **launcher.timing._last_train})
    _save_state(state)

    if train_exc is not None:
        raise train_exc
