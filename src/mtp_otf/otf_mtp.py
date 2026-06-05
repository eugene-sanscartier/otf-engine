import concurrent.futures
import json
import os

import numpy

import ase
import ase.io.lammpsrun

from .io_cfg import read_cfg, write_cfg
from .launchers import NestedMPILauncher

try:
    from .mtp_backend import (calculate_grade as _calc_grade_py, select_add as _select_add_py, train as _train_py)
    _BUILTIN = True
except ImportError:
    _BUILTIN = False

OTF_STATE_FILE = "otf_state.json"


def _load_state():
    if os.path.isfile(OTF_STATE_FILE):
        with open(OTF_STATE_FILE, "r") as f:
            return json.load(f)
    return {}


def _save_state(state):
    with open(OTF_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def preselected_dump2cfg(extrapolative_dumps, extrapolative_candidates_cfg, extrapolation_field="f_extrapolation_grade"):
    collected_dumps = []
    for extrapolative_dump in extrapolative_dumps:
        with open(extrapolative_dump) as dump_file:
            dumps = ase.io.lammpsrun.read_lammps_dump_text(dump_file, index=slice(None))
            print("Reading extrapolative dump : ", extrapolative_dump, " with ", len(dumps), " structures")

            if len(dumps) > 100:
                print("Warning: Large extrapolative dump with ", len(dumps), " structures, this may cause performance issues.")
                dumps = dumps[-100:]

            collected_dumps += dumps

        try:
            os.remove(extrapolative_dump)
        except Exception as e:
            print(f"Warning: {extrapolative_dump}: {e}")

    for dump in collected_dumps:
        if dump.has(extrapolation_field):
            dump.set_array("nbh_grades", dump.get_array(extrapolation_field).flatten())

    try:
        with open(extrapolative_candidates_cfg, mode="w") as preselected_file:
            write_cfg(preselected_file, collected_dumps)
    except Exception as e:
        print(f"Error: Exception {extrapolative_candidates_cfg}: {e}")


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


def _load_evaluator():
    import importlib.util
    path = os.path.join(os.getcwd(), "evaluator.py")
    spec = importlib.util.spec_from_file_location("evaluator", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.evaluator


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


def preselected_filter(cfgs, gamma_tolerance, gamma_max, gamma_max0_cap, extreme_lock_after_ntimes=10):

    state = _load_state()
    gamma_max0 = state.get("gamma_max0", gamma_max0_cap)

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

    if not numpy.any(gammas < gamma_max) and numpy.any(gammas < gamma_max0_cap):
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


def load_structures(set_name):
    with open(set_name, mode="r") as set_file:
        cfgs = read_cfg(set_file)
    return cfgs


def save_structures(set_name, cfgs):
    with open(set_name, mode="w") as set_file:
        write_cfg(set_file, cfgs)


def _eval_one(i, structure, evaluator_fn, launcher, env, force_threshold):
    """Evaluate one structure; return evaluated Atoms or None on failure/threshold."""
    eval_dir = f"eval_{i:03d}"
    print(f"Calculating structure {i + 1}")
    try:
        result = launcher.call_evaluator(evaluator_fn, structure, eval_dir, env)
        if force_threshold is not None and forcesthr_excess(result, threshold=force_threshold):
            print(f"Warning: Structure {i + 1} has forces exceeding threshold, skipping.")
            print(f" Max force component: {numpy.max(numpy.abs(numpy.array(result.calc.results['forces']))):.2f}")
            return None
        return result
    except Exception as e:
        print(f"Error evaluating structure {i + 1}: {e}")
        print("Warning: Error in eval_structures")
        try:
            espresso_err = os.path.join(eval_dir, "espresso.err")
            print("Output of espresso.err")
            with open(espresso_err) as err_file:
                print(err_file.read())
            import shutil
            shutil.rmtree(os.path.join(eval_dir, "pwscf.save"))
        except Exception as e2:
            print(f"Warning: Could not clean up Espresso artifacts: {e2}")
        return None


def eval_structures(selected_extrapolative, training_set, evaluator_fn, launcher, env, force_threshold=None):
    with open(selected_extrapolative, mode="r") as selected_file:
        selected_structures = read_cfg(selected_file)

    with open(training_set, mode="r") as training_file:
        training_structures = read_cfg(training_file)

    if launcher.parallel_eval and len(selected_structures) > 1:
        print(f"Evaluating {len(selected_structures)} structures in parallel.")
        results = [None] * len(selected_structures)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(_eval_one, i, struct, evaluator_fn, launcher, env, force_threshold): i for i, struct in enumerate(selected_structures)}
            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                try:
                    results[i] = future.result()
                except Exception as e:
                    print(f"Error in parallel eval of structure {i + 1}: {e}")

        successful = [r for r in results if r is not None]
        if successful:
            training_structures += successful
            with open(training_set, mode="w") as training_file:
                write_cfg(training_file, training_structures)
    else:
        for i, selected_structure in enumerate(selected_structures):
            print(f"Calculating structure {i + 1}/{len(selected_structures)}")
            result = _eval_one(i, selected_structure, evaluator_fn, launcher, env, force_threshold)
            if result is not None:
                training_structures.append(result)
                with open(training_set, mode="w") as training_file:
                    write_cfg(training_file, training_structures)


def main(args, _env, launcher=None, mlp_command=None, train_n_procs=None, evaluator_fn=None, use_builtin=True):
    # Resolve mlp_command: explicit param > env var > error
    if _BUILTIN and use_builtin:
        mlp_command = None  # not needed for mlp steps
    else:
        if mlp_command is None:
            mlp_command = os.environ.get("OTF_MTP_COMMAND")
        if not mlp_command:
            raise RuntimeError("mlp_command not provided and OTF_MTP_COMMAND environment variable is not set. Pass mlp_command= or set: export OTF_MTP_COMMAND=/path/to/mlp")

    if launcher is None:
        launcher = NestedMPILauncher()

    if evaluator_fn is None:
        evaluator_fn = _load_evaluator()

    extrapolative_candidates = "preselected.cfg"
    extrapolative_candidates_out = "preselected"
    selected_extrapolative = "selected.cfg"
    extrapolation_field = "f_extrapolation_grade"

    exit_returncode = 0

    preselected_dump2cfg(args.extrapolative_dumps, extrapolative_candidates, extrapolation_field)

    if args.preselection_filtering:
        # failsafe: lammps extrapolation fix-halt sometimes stops before grade calculation
        try:
            if _BUILTIN and use_builtin:
                cfgs = load_structures(extrapolative_candidates)
                cfgs = _calc_grade_py(args.potential, cfgs)
                save_structures(extrapolative_candidates, cfgs)
            else:
                launcher.run(f"{mlp_command} calculate_grade {args.potential} {extrapolative_candidates} {extrapolative_candidates_out}.calculate_grade", "mlip_calculate_grade.log", _env, n_procs=1)
                os.replace(f"{extrapolative_candidates_out}.calculate_grade.0", extrapolative_candidates)
        except Exception as e:
            print(f"calculate_grade failed: {e}")
            exit_returncode = getattr(e, 'returncode', 1)

        cfgs = load_structures(extrapolative_candidates)
        filtred_cfgs = preselected_filter(cfgs, args.gamma_tolerance, args.gamma_max, args.gamma_max0_cap, extreme_lock_after_ntimes=args.extreme_lock_after_ntimes)
        save_structures(extrapolative_candidates, filtred_cfgs)

    if args.max_structures > 0:
        cfgs = load_structures(extrapolative_candidates)
        filtred_cfgs = max_structureselection(cfgs, max_structures=args.max_structures)
        save_structures(extrapolative_candidates, filtred_cfgs)

    try:
        if _BUILTIN and use_builtin:
            train_cfgs = load_structures(args.training_set)
            candidate_cfgs = load_structures(extrapolative_candidates)
            selected, _sel_weights, _sel_A, _sel_invA = _select_add_py(args.potential, train_cfgs, candidate_cfgs)
            save_structures(selected_extrapolative, selected)
        else:
            launcher.run(
                f"{mlp_command} select_add {args.potential} {args.training_set} {extrapolative_candidates} {selected_extrapolative}",
                "mlip_select_add.log",
                _env,
                n_procs=1,
            )
    except Exception as e:
        print(f"select_add failed: {e}")
        exit_returncode = getattr(e, 'returncode', 1)

    eval_structures(selected_extrapolative, args.training_set, evaluator_fn, launcher, _env, force_threshold=args.force_threshold)

    try:
        if _BUILTIN and use_builtin:
            train_cfgs = load_structures(args.training_set)
            _train_py(args.potential, train_cfgs, f"tmp_{args.potential}", args.iteration_limit)
        else:
            launcher.run(f"{mlp_command} train {args.potential} {args.training_set} --save_to=tmp_{args.potential} --iteration_limit={args.iteration_limit} ", "mlip_train.log", _env, n_procs=train_n_procs)
        os.replace(f"tmp_{args.potential}", args.potential)
    except Exception as e:
        print(f"train failed: {e}")
        exit_returncode = getattr(e, 'returncode', 1)

    return exit_returncode
