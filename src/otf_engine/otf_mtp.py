import concurrent.futures
import json
import os
import shutil
import traceback
from pathlib import Path

import numpy

import ase
import ase.io.lammpsrun

from .io_cfg import read_cfg, write_cfg
from .mtp_backend import calculate_grade, select_add
from .cycles import current_cycle_dir
from .launchers import Launcher

OTF_STATE_FILE = "otf_state.json"


def _load_state():
    if os.path.isfile(OTF_STATE_FILE):
        with open(OTF_STATE_FILE, "r") as f:
            return json.load(f)
    return {}


def _save_state(state):
    with open(OTF_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def load_extrapolative_dumps(extrapolative_dumps, extrapolation_field="f_extrapolation_grade", species=None):
    collected_dumps = []
    for extrapolative_dump in extrapolative_dumps:
        with open(extrapolative_dump) as dump_file:
            dumps = ase.io.lammpsrun.read_lammps_dump_text(dump_file, index=slice(None), specorder=species)
            print("Reading extrapolative dump : ", extrapolative_dump, " with ", len(dumps), " structures")

            if len(dumps) > 100:
                print("Warning: Large extrapolative dump with ", len(dumps), " structures, this may cause performance issues.")
                _indices = numpy.random.choice(len(dumps), size=100, replace=False)
                dumps = [dumps[i] for i in _indices]

            collected_dumps += dumps

        # try:
        #     os.remove(extrapolative_dump)
        # except Exception as e:
        #     print(f"Warning: {extrapolative_dump}: {e}")

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
    return gamma_max0_new


def forcesthr_excess(atoms, threshold):
    if atoms.calc is None or "forces" not in atoms.calc.results:
        return False
    return numpy.max(numpy.abs(atoms.calc.results["forces"])) > threshold


def _record_non_extreme(state, extreme_lock_after_ntimes):
    state["non_extreme_count"] = state.get("non_extreme_count", 0) + 1
    if state["non_extreme_count"] >= extreme_lock_after_ntimes:
        state["extreme_allowed"] = False
    _save_state(state)


def _checkgrade(cfg):
    if "nbh_grades" in cfg.arrays:
        return cfg.arrays["nbh_grades"].max()
    if "features" in cfg.info and "MV_grade" in cfg.info["features"]:
        return cfg.info["features"]["MV_grade"]
    return 0


def preselected_filter(cfgs, gamma_tolerance, gamma_max, gamma_max_cap, extreme_lock_after_ntimes=10):

    state = _load_state()
    gamma_max0 = state.get("gamma_max0", gamma_max_cap)

    print("Preselected structures count: ", len(cfgs))

    gammas = numpy.array([_checkgrade(cfg) for cfg in cfgs])
    mask = gammas > gamma_tolerance
    cfgs = [cfg for cfg, m in zip(cfgs, mask) if m]
    gammas = gammas[mask]

    if not cfgs:
        print(f"No structures above gamma_tolerance={gamma_tolerance:.4f} — nothing to select.")
        return []

    filtred_cfgs = []

    if numpy.any(gammas < gamma_max):
        filtred_cfgs = [cfg for cfg, g in zip(cfgs, gammas) if g < gamma_max]
        _record_non_extreme(state, extreme_lock_after_ntimes)

    elif numpy.any(gammas < gamma_max0):
        print(f"gamma_max0 = {gamma_max0:.4f} (history length = {len(state.get('gamma_max0_history', []))})")
        idx = numpy.argmin(gammas)
        filtred_cfgs = [cfgs[idx]]
        print(f"Selected structure with gamma = {gammas[idx]}")
        _record_non_extreme(state, extreme_lock_after_ntimes)

    else:
        extreme_allowed = state.get("extreme_allowed", True)
        non_extreme_count = state.get("non_extreme_count", 0)
        state["extreme_count"] = state.get("extreme_count", 0) + 1
        print(f"Extreme Warning: all gammas > gamma_max0={gamma_max0:.4f}. "
              f"min gamma = {numpy.min(gammas):.4f}, "
              f"non_extreme_count={non_extreme_count} (extreme_lock_after_ntimes={extreme_lock_after_ntimes}), "
              f"extreme_allowed={extreme_allowed}")
        if extreme_allowed:
            filtred_cfgs = [cfgs[numpy.argmin(gammas)]]
            state["non_extreme_count"] = 0
            print(f"Selecting structure with gamma = {numpy.min(gammas):.4f}")
        else:
            print(f"Skipping selection: {non_extreme_count} consecutive non-extreme iterations reached limit of {extreme_lock_after_ntimes}")
        _save_state(state)

    if not numpy.any(gammas < gamma_max) and numpy.any(gammas < gamma_max_cap):
        gamma_max0_new = _update_gamma_max0(state, numpy.min(gammas), gamma_max)
        print(f"Updated gamma_max0: {gamma_max0:.4f} -> {gamma_max0_new:.4f}")
        _save_state(state)

    print("Post-Preselection filtered structures count: ", len(filtred_cfgs))

    return filtred_cfgs


def max_structureselection(filtred_cfgs, max_structures=-1):

    if max_structures > 0 and len(filtred_cfgs) > max_structures:
        rnd_selected = numpy.random.choice(len(filtred_cfgs), size=max_structures, replace=False)
        filtred_cfgs = [filtred_cfgs[i] for i in rnd_selected]
        print("Post-Preselection max-structures count: ", len(filtred_cfgs))

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
            max_f = numpy.max(numpy.abs(numpy.array(result.calc.results['forces'])))
            return None, f"skipped (max force {max_f:.2f} eV/Å)"
        return result, "ok"
    except Exception as e:
        eval_dir.mkdir(parents=True, exist_ok=True)
        with open(eval_dir / "eval.log", "a") as _f:
            traceback.print_exc(file=_f)
        try:
            print(f"[struct {i+1}] espresso.err:\n{(eval_dir / 'espresso.err').read_text()}")
            shutil.rmtree(eval_dir / "pwscf.save")
        except Exception:
            pass
        return None, f"failed: {e}"


def eval_structures(selected_structures, training_set, evaluator_fn, launcher, force_threshold=None):
    n = len(selected_structures)
    w = len(str(n))
    parallel = launcher.concurrent_eval and n > 1
    print(f"Evaluating {n} structures {'concurrently' if parallel else 'sequentially'}.")
    n_ok = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=None if parallel else 1) as executor:
        futures = {executor.submit(_eval_one, i, s, evaluator_fn, launcher, force_threshold): i for i, s in enumerate(selected_structures)}
        for k, future in enumerate(concurrent.futures.as_completed(futures), 1):
            i = futures[future]
            result, status = future.result()
            print(f"[{k:{w}d}/{n}] struct {i+1:{w}d} — {status}")
            if result is not None:
                n_ok += 1
                save_structures(training_set, [result], append=True)
    print(f"Evaluated {n_ok}/{n} successfully.")
    return n_ok


def main(args, launcher:Launcher=None, mlp_command=None, evaluator_fn=None):
    """Run one OTF-MTP update cycle from extrapolative dumps to a retrained model.

    The flow is: load and clean the candidate pool, select which structures
    should extend the training set, evaluate those structures with the prepared
    backend, then retrain the potential in place.
    """

    # Step 1: load the extrapolative structures emitted by the upstream run.
    # These dumps are the raw candidate pool from which new training structures
    # may be chosen.
    candidate_structures = load_extrapolative_dumps(args.extrapolative_dumps, species=args.species)

    # Step 2: ensure every candidate carries an extrapolation grade, even when
    # LAMMPS stopped early and the dump does not already contain the final or correct
    # extrapolation metadata needed by the downstream selection logic.
    candidate_structures = calculate_grade(args.potential, candidate_structures)

    # Step 3: optionally apply the preselection policy so uninteresting
    # or disallowed candidates are removed before selection and evaluation stages.
    if args.preselection_filtering:
        candidate_structures = preselected_filter(candidate_structures, args.gamma_tolerance, args.gamma_max, args.gamma_max_cap, extreme_lock_after_ntimes=args.extreme_lock_after_ntimes)

    # Step 4: optionally cap the surviving pool size. This keeps the next stages
    # bounded when the extrapolative search produced many eligible structures.
    if args.max_structures > 0:
        candidate_structures = max_structureselection(candidate_structures, max_structures=args.max_structures)

    # Step 5: run the structure-selection step.
    train_structures = load_structures(args.training_set, args.species)
    selected_structures, _ = select_add(args.potential, train_structures, candidate_structures)

    # Step 6: evaluate the selected structures with the configured backend and
    # write evaluated structure into the training set for the retraining step.
    n_ok = eval_structures(selected_structures, args.training_set, evaluator_fn, launcher, force_threshold=args.force_threshold) if selected_structures else 0
    if not n_ok: return print("No configurations selected or evaluated — skipping training.")

    # Step 7: retrain the potential on the updated training set.
    launcher.run(f"{mlp_command} train {args.potential} {args.training_set} --save_to=tmp_{args.potential} --iteration_limit={args.iteration_limit} ", log_file="mlip_train.log")
    os.replace(f"tmp_{args.potential}", args.potential)

