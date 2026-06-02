import os
import argparse
from .otf_mtp import main as _main


def main():
    parser = argparse.ArgumentParser(prog=None, description="Utility to select structures for training se based on D-optimality criterion")

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

    args = parser.parse_args()

    # // Base env related to OMPI_
    SAVE_ENVNAME = [
        "OMPI_MCA_btl",
        "OMPI_MCA_pml",
        "OMPI_MCA_mtl",
        "OMPI_MCA_coll",
        "OMPI_MCA_mpi_oversubscribe",
        "OMPI_MCA_pmix",
    ]

    DELETE_ENVNAME = [
        "OMPI_MCA_ess",
        "OMPI_MCA_ess_base_jobid",
        "OMPI_UNIVERSE_SIZE",
    ]

    # Remove all OMPI_ environment variables to avoid issues with MPI
    save_env = os.environ.copy()

    # del_env = {k: os.environ.pop(k) for k, v in os.environ.items() if k.startswith("OMPI_") and k not in SAVE_ENVNAME}
    del_env = {k: os.environ.pop(k) for k, v in os.environ.items() if k.startswith("OMPI_") and k in DELETE_ENVNAME}
    # print("Removed OMPI_ environment variables: ", del_env)

    returncode = _main(args, os.environ)

    os.environ = save_env

    if returncode != 0:
        print(f"One program exited with return code: {returncode}")
        print("Exiting with return code 0.")

    exit(0)


if __name__ == "__main__":
    main()
