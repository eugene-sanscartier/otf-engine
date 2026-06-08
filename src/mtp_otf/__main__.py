import os
import sys
import argparse
from .otf_mtp import main as _main
from .launchers import NestedLauncher, ForkLauncher, SlurmLauncher
from .cycles import next_cycle_dir, archive_cycle


def _load_evaluator():
    import importlib.util
    path = os.path.join(os.getcwd(), "evaluator.py")
    spec = importlib.util.spec_from_file_location("evaluator", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.evaluator


def main():
    parser = argparse.ArgumentParser(prog=None, description="Utility to select structures for training set based on D-optimality criterion")

    parser.add_argument("extrapolative_dumps", nargs='+', help=" extrapolative_structures.dump", type=str)
    parser.add_argument("-p", "--potential", help="input potential file name, will override input file 'potential' section", type=str, default="potential.almtp")
    parser.add_argument("-t", "--training_set", help="Training dataset file name, ex.: train.cfg", type=str, default="train.cfg")

    parser.add_argument("-P", "--no_preselection_filtering", help="Preselection filtering", dest='preselection_filtering', action='store_false')

    parser.add_argument("-g", "--gamma_tolerance", help="Gamma tolerance", default=1.010, type=float)
    parser.add_argument("-G", "--gamma_max", help="Gamma max", default=0, type=float)
    parser.add_argument("-D", "--gamma_max_cap", help="Gamma max_0 cap (initial value; rolling update never fires above this)", default=10000, type=float)
    parser.add_argument("-X", "--extreme_lock_after_ntimes", help="After n cycle without extreme extrapolation configuration only, no more extreme extrapolation configuration are selected.", default=5, type=int)

    parser.add_argument("-m", "--max_structures", help="Max structures selection", default=-1, type=int)
    parser.add_argument("-l", "--iteration_limit", help="Number of maximum iteration in training algorithm", default=300, type=int)
    parser.add_argument("-f", "--force_threshold", help="Force threshold (eV/Å): structures with max force component exceeding this value are skipped. Default: no threshold.", default=None, type=float)

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

    evaluator_fn = _load_evaluator()
    os.environ["COMMAND_PREFIX"] = launcher.command_prefix()

    cycle_dir = next_cycle_dir()

    try:
        _main(args, launcher=launcher, mlp_command=mlp_command, evaluator_fn=evaluator_fn)
    except Exception as e:
        print(f"Error during execution: {e}", file=sys.stderr)
        # archive_cycle(cycle_dir, args.potential, args.training_set, dump_files=args.extrapolative_dumps)
        # sys.exit(1)

    archive_cycle(cycle_dir, args.potential, args.training_set, dump_files=args.extrapolative_dumps)
    sys.exit(0)


if __name__ == "__main__":
    main()
