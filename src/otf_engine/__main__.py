import importlib.util
import inspect
import logging
import os
import re
import sys
import argparse
from pathlib import Path
from .otf_mtp import OTFCycle, run_cycle
from .launchers import NestedLauncher, ForkLauncher, SlurmLauncher
from .cycles import next_cycle_dir, LOG_FILE, LOG_FORMAT

logger = logging.getLogger(__name__)


def _cgroup_cpus() -> set[int]:
    """Return the CPUs this process's cgroup v2 cpuset allows."""
    cgroup = Path("/proc/self/cgroup").read_text().split("0::", 1)[1].split()[0]
    text = Path(f"/sys/fs/cgroup{cgroup}/cpuset.cpus.effective").read_text()
    bounds = [[int(n) for n in part.split("-")] for part in text.strip().split(",")]
    cpus = {cpu for b in bounds for cpu in range(b[0], b[-1] + 1)}
    return cpus


def main():

    parser =argparse.ArgumentParser(prog=None, description="Utility to select structures for training set based on D-optimality criterion")
    # The cycle's options take their defaults from OTFCycle.
    defaults = {name: parameter.default for name, parameter in inspect.signature(OTFCycle).parameters.items()}

    parser.add_argument("--extrapolative_dumps", nargs='+', required=True, metavar="DUMP", dest="extrapolative_dumps", help="Extrapolative dump files (glob patterns allowed).", type=str)
    parser.add_argument("-p", "--potential", help="input potential file name (default: %(default)s)", type=str, default=defaults["potential"])
    parser.add_argument("-t", "--training_set", help="Training dataset file name (default: %(default)s)", type=str, default=defaults["training_set"])

    parser.add_argument("-P", "--no_preselection_filtering", help="Preselection filtering", dest='preselection_filtering', action='store_false', default=defaults["preselection_filtering"])

    parser.add_argument("-g", "--gamma_tolerance", help="Gamma tolerance (default: %(default)s)", default=defaults["gamma_tolerance"], type=float)
    parser.add_argument("-G", "--gamma_max", help="Gamma max (default: %(default)s)", default=defaults["gamma_max"], type=float)
    parser.add_argument("-D", "--gamma_max_cap", help="Gamma max_0 cap (initial value; rolling update never fires above this) (default: %(default)s)", default=defaults["gamma_max_cap"], type=float)
    parser.add_argument("-X", "--extreme_lock_after_ntimes", help="After n cycle without extreme extrapolation configuration only, no more extreme extrapolation configuration are selected (default: %(default)s).", default=defaults["extreme_lock_after_ntimes"], type=int)

    parser.add_argument("-m", "--max_structures", help="Max structures selection (default: %(default)s, no cap)", default=defaults["max_structures"], type=int)
    parser.add_argument("-l", "--iteration_limit", help="Number of maximum iteration in training algorithm (default: %(default)s)", default=defaults["iteration_limit"], type=int)
    parser.add_argument("-f", "--force_threshold", help="Force threshold (eV/Å): structures with max force component exceeding this value are skipped. Default: no threshold.", default=defaults["force_threshold"], type=float)
    parser.add_argument("-s", "--species", nargs='+', type=str, default=defaults["species"], metavar="SYMBOLS", help="Ordered element symbols matching MTP type indices. Must cover ALL types defined in the potential and training set, not just those present in the current simulation. Space-separated: -s Al Cu, or single-string: -s Al,Cu or -s '[Al, Cu]'.")

    parser.add_argument("--launcher", choices=["nested", "fork", "slurm"], default="nested", help="Execution backend. 'nested' (default): wrap calls with mpirun. "
                        "'fork': run binary directly in MPI universe. "
                        "'slurm': submit each call as a batch job via sbatch --wait.")
    parser.add_argument("--batch-args", default="", type=str, metavar="ARGS", help="Extra batch-scheduler options as one shell string (batch launchers only). "
                        "For slurm: raw sbatch options (e.g. --batch-args='--partition=gpu --time=01:00:00').")
    parser.add_argument("--runner-args", default="", type=str, metavar="ARGS", help="Extra arguments appended to the runner executable as one shell string. "
                        "For nested: appended to mpirun (e.g. --runner-args='--oversubscribe'). "
                        "For slurm: appended to srun when used as COMMAND_PREFIX or for sequential mlp calls "
                        "(e.g. --runner-args='--bind-to core'). Ignored for fork.")
    parser.add_argument("--sequential-eval", dest="concurrent_eval", action="store_false", help="Evaluate structures sequentially instead of concurrently for the slurm launcher.")
    parser.set_defaults(concurrent_eval=True)
    args = parser.parse_args()
    if args.species:
        args.species = [sym for s in args.species for sym in re.findall(r'[A-Z][a-z]*', s)]

    mlp_command = os.environ.get("OTF_MTP_COMMAND")
    if not mlp_command:
        raise RuntimeError("mlp_command not provided and OTF_MTP_COMMAND environment variable is not set. Pass mlp_command= or set: export OTF_MTP_COMMAND=/path/to/mlp")

    match args.launcher:
        case "nested":
            launcher = NestedLauncher(runner_args=args.runner_args)
        case "fork":
            launcher = ForkLauncher()
        case "slurm":
            launcher = SlurmLauncher(batch_args=args.batch_args, concurrent_eval=args.concurrent_eval, runner_args=args.runner_args)

    spec = importlib.util.spec_from_file_location("evaluator", "evaluator.py")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    os.environ["COMMAND_PREFIX"] = launcher.command_prefix()

    cycle_dir = next_cycle_dir()
    log_path = cycle_dir / LOG_FILE

    # For records from outside the package: the cycle logs the package's there itself.
    logging.basicConfig(level=logging.INFO, filename=log_path, filemode="a", format=LOG_FORMAT)
    print(f"{cycle_dir.name} running — {log_path}")

    # The engine is often started by one MPI rank (pyKMC) and would inherit its binding to one core.
    try:
        os.sched_setaffinity(0, _cgroup_cpus())
    except Exception as e:
        logger.warning(f"Inherited {os.process_cpu_count()} CPUs, cgroup cpuset unreadable: {e!r}")

    options = vars(args)
    for name in ("launcher", "batch_args", "runner_args", "concurrent_eval"): del options[name]
    try:
        run_cycle(launcher=launcher, evaluator_fn=evaluator.evaluator, mlp_command=mlp_command, cycle_dir=cycle_dir, **options)
    except Exception:
        sys.exit(67)
    sys.exit(0)


if __name__ == "__main__":
    main()
