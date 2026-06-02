import json
import os
import subprocess

import numpy

import ase
import ase.io.lammpsrun

from .io_cfg import read_cfg, write_cfg

mlp = os.environ["OTF_MTP_COMMAND"]
if mlp == "":
    print("Error OTF_MTP_COMMAND variable not set, set with export OTF_MTP_COMMAND=\"/path/to/mlp\" (in bash) before this script")


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
    return atoms.calc is not None and "forces" in atoms.calc.results and numpy.max(numpy.abs(numpy.array(atoms.calc.results["forces"]))) > threshold


def _run_mlp(cmd, log_path, env):
    """Run a shell command string, append output to log_path. Raises on failure."""
    print(f"running: {cmd}")
    with open(log_path, "a") as f:
        subprocess.run(cmd, shell=True, text=True, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)


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
    cfgs = [cfgs[i] for i, m in enumerate(mask) if m]
    gammas = gammas[mask]

    if not cfgs:
        print(f"No structures above gamma_tolerance={gamma_tolerance:.4f} — nothing to select.")
        return []

    filtred_cfgs = []

    if numpy.any(gammas < gamma_max):
        filtred_cfgs = [cfgs[i] for i in numpy.where(gammas < gamma_max)[0]]
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


def eval_structures(selected_extrapolative, training_set, evaluator_fn, force_threshold=None):
    with open(selected_extrapolative, mode="r") as selected_file:
        selected_structures = read_cfg(selected_file)

    with open(training_set, mode="r") as training_file:
        training_structures = read_cfg(training_file)

    for i, selected_structure in enumerate(selected_structures):
        print(f"Calculating structure {i+1}/{len(selected_structures)}")

        try:
            selected_structure = evaluator_fn(selected_structure)
            if force_threshold is not None and forcesthr_excess(selected_structure, threshold=force_threshold):
                print(f"Warning: Structure {i+1} has forces exceeding threshold after evaluation, skipping addition to training set.")
                print(f" Max force component: {numpy.max(numpy.abs(numpy.array(selected_structure.calc.results['forces']))):.2f}")
                continue

            training_structures += [selected_structure]
            with open(training_set, mode="w") as training_file:
                write_cfg(training_file, training_structures)
        except Exception as e:
            print(f"Error evaluating structure {i+1}: {e}")
            print("Warning: Error in eval_structures")
            try:
                print("Output of espresso.err")
                with open("espresso.err", mode="r") as err_file:
                    print(err_file.read())

                print("Trying to remove pwscf.save directory")
                import shutil
                shutil.rmtree("pwscf.save")
            except Exception as e:
                print(f"Warning: Could not remove pwscf.save directory: {e}")

    return 0


def main(args_parse, _env):
    evaluator_fn = _load_evaluator()

    potential = args_parse.potential
    training_set = args_parse.training_set
    extrapolative_dumps = args_parse.extrapolative_dumps
    extrapolative_candidates = "preselected.cfg"
    extrapolative_candidates_out = "preselected"
    selected_extrapolative = "selected.cfg"
    extrapolation_field = "f_extrapolation_grade"

    preselection_filtering = args_parse.preselection_filtering
    gamma_tolerance = args_parse.gamma_tolerance
    gamma_max = args_parse.gamma_max
    gamma_max0_cap = args_parse.gamma_max0_cap
    extreme_lock_after_ntimes = args_parse.extreme_lock_after_ntimes
    max_structures = args_parse.max_structures
    iteration_limit = args_parse.iteration_limit
    force_threshold = args_parse.force_threshold

    exit_returncode = 0

    preselected_dump2cfg(extrapolative_dumps, extrapolative_candidates, extrapolation_field)

    if preselection_filtering:
        # failsafe: lammps extrapolation fix-halt sometimes stops before grade calculation
        try:
            _run_mlp(
                f"mpirun -n 1 {mlp} calculate_grade {potential} {extrapolative_candidates} {extrapolative_candidates_out}.calculate_grade",
                "mlip_calculate_grade.log", _env)
            os.replace(f"{extrapolative_candidates_out}.calculate_grade.0", extrapolative_candidates)
        except Exception as e:
            print(f"calculate_grade failed: {e}")
            exit_returncode = getattr(e, 'returncode', 1)

        cfgs = load_structures(extrapolative_candidates)
        filtred_cfgs = preselected_filter(cfgs, gamma_tolerance, gamma_max, gamma_max0_cap, extreme_lock_after_ntimes=extreme_lock_after_ntimes)
        save_structures(extrapolative_candidates, filtred_cfgs)

    if max_structures > 0:
        cfgs = load_structures(extrapolative_candidates)
        filtred_cfgs = max_structureselection(cfgs, max_structures=max_structures)
        save_structures(extrapolative_candidates, filtred_cfgs)

    try:
        _run_mlp(
            f"mpirun -n 1 {mlp} select_add {potential} {training_set} {extrapolative_candidates} {selected_extrapolative}",
            "mlip_select_add.log", _env)
    except Exception as e:
        print(f"select_add failed: {e}")
        exit_returncode = getattr(e, 'returncode', 1)

    eval_structures(selected_extrapolative, training_set, evaluator_fn, force_threshold=force_threshold)

    try:
        _run_mlp(
            f"mpirun {mlp} train {potential} {training_set} --save_to=tmp_{potential} --iteration_limit={iteration_limit} --al_mode=nbh",
            "mlip_train.log", _env)
        os.replace(f"tmp_{potential}", potential)
    except Exception as e:
        print(f"train failed: {e}")
        exit_returncode = getattr(e, 'returncode', 1)

    return exit_returncode
