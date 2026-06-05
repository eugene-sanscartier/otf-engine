import os
import sys
import argparse
from .otf_mtp import main as _main
from .launchers import NestedMPILauncher, ForkLauncher, SlurmLauncher


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
    parser.add_argument("-D", "--gamma_max0_cap", help="Gamma max_0 cap (initial value; rolling update never fires above this)", default=10000, type=float)
    parser.add_argument("-X", "--extreme_lock_after_ntimes", help="After n cycle without extreme extrapolation configuration only, no more extreme extrapolation configuration are selected.", default=5, type=int)

    parser.add_argument("-m", "--max_structures", help="Max structures selection", default=-1, type=int)
    parser.add_argument("-l", "--iteration_limit", help="Number of maximum iteration in training algorithm", default=300, type=int)
    parser.add_argument("-f", "--force_threshold", help="Force threshold (eV/Å): structures with max force component exceeding this value are skipped. Default: no threshold.", default=None, type=float)

    parser.add_argument("--launcher", choices=["nested", "fork", "slurm"], default="nested", help="Execution backend. 'nested' (default): wrap calls with mpirun. "
                        "'fork': run binary directly in MPI universe. "
                        "'slurm': submit each call as a batch job via sbatch --wait.")
    parser.add_argument("--launcher-extra", default="", type=str, metavar="ARGS", help="Extra launcher arguments as one shell string. "
                        "For nested: inserted between -n and the mlp binary "
                        "(e.g. --launcher-extra='--oversubscribe'). "
                        "For fork: accepted but ignored. "
                        "For slurm: passed as raw sbatch options "
                        "(e.g. --launcher-extra='--partition=gpu --time=01:00:00').")
    parser.add_argument("--sequential-eval", dest="parallel_eval", action="store_false", help="Evaluate structures sequentially instead of in parallel for the slurm launcher.")
    parser.set_defaults(parallel_eval=True)

    args = parser.parse_args()

    mlp_command = os.environ.get("OTF_MTP_COMMAND")
    if not mlp_command:
        raise RuntimeError("mlp_command not provided and OTF_MTP_COMMAND environment variable is not set. Pass mlp_command= or set: export OTF_MTP_COMMAND=/path/to/mlp")

    extra_args = args.launcher_extra
    if args.launcher == "nested":
        launcher = NestedMPILauncher(mpirun_args=extra_args)
    elif args.launcher == "fork":
        launcher = ForkLauncher(fork_args=extra_args)
    elif args.launcher == "slurm":
        launcher = SlurmLauncher(sbatch_args=extra_args, parallel_eval=args.parallel_eval)
    else:
        print(f"Error: unknown launcher {args.launcher!r}", file=sys.stderr)
        sys.exit(1)

    if launcher is None:
        launcher = NestedMPILauncher()

    evaluator_fn = _load_evaluator()

    try:
        _main(args, os.environ, launcher=launcher, mlp_command=mlp_command, evaluator_fn=evaluator_fn)
    except Exception as e:
        print(f"Error during execution: {e}", file=sys.stderr)

    print("Exiting with return code 0.")
    sys.exit(0)


if __name__ == "__main__":
    main()
