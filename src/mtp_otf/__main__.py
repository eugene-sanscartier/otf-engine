import os
import sys
import argparse
from .otf_mtp import main as _main, _BUILTIN
from .launchers import NestedMPILauncher, ForkLauncher, SlurmLauncher


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
    parser.add_argument("--train-n-procs", default=None, type=int, metavar="N", help="Number of MPI processes for the train step. "
                        "Default: None (use all available, i.e. omit -n from mpirun). "
                        "Only used by nested (mpirun) and slurm launchers.")
    parser.add_argument("--launcher-extra", nargs="*", default=None, metavar="ARG", help="Extra arguments for the launcher. "
                        "For nested: passed verbatim between -n and the mlp binary "
                        "(e.g. --launcher-extra --oversubscribe). "
                        "For slurm: passed as sbatch options "
                        "(e.g. --launcher-extra --partition=gpu --time=01:00:00).")
    parser.add_argument("--eval-n-procs", default=None, type=int, metavar="N", help="Number of processes per structure evaluation (fork launcher only). "
                        "Default: None (use all available CPUs). "
                        "The evaluator profile command must be set to just the binary path "
                        "(e.g. 'pw.x'), not 'mpiexec -n N pw.x'.")
    parser.add_argument("--no-parallel-eval", dest="parallel_eval", action="store_false", help="Evaluate structures sequentially instead of in parallel. "
                        "Default is parallel (applies to slurm and fork launchers).")
    parser.set_defaults(parallel_eval=True)

    args = parser.parse_args()

    mlp_command = os.environ.get("OTF_MTP_COMMAND")
    if not mlp_command and not _BUILTIN:
        print("Error: OTF_MTP_COMMAND environment variable is not set"
              "Set OTF_MTP_COMMAND=/path/to/mlp ", file=sys.stderr)
        sys.exit(1)

    extra_args = list(args.launcher_extra) if args.launcher_extra else []
    if args.launcher == "nested":
        launcher = NestedMPILauncher(extra_args=" ".join(extra_args))
    elif args.launcher == "fork":
        launcher = ForkLauncher(eval_n_procs=args.eval_n_procs, parallel_eval=args.parallel_eval)
    elif args.launcher == "slurm":
        launcher = SlurmLauncher(sbatch_args=extra_args, parallel_eval=args.parallel_eval)
    else:
        print(f"Error: unknown launcher {args.launcher!r}", file=sys.stderr)
        sys.exit(1)

    returncode = _main(args, os.environ, launcher=launcher, mlp_command=mlp_command, train_n_procs=args.train_n_procs)

    if returncode != 0:
        print(f"One program exited with return code: {returncode}")
        print("Exiting with return code 0.")

    sys.exit(0)


if __name__ == "__main__":
    main()
