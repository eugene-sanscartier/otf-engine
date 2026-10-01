import os
import argparse
from pathlib import Path

import ase.io
import ase.io.extxyz

import ase.calculators
import ase.calculators.espresso
from ase.calculators.espresso import EspressoProfile


def build_command(binary, ase_env=None):
    """Assemble the full executable command from env vars set by the kmtp-otf launcher.

    The launcher writes ``COMMAND_PREFIX`` before calling evaluator() (or before
    submitting this script as a batch job via SlurmLauncher):

        COMMAND_PREFIX — prefix produced by launcher.command_prefix():
                         ``"mpirun [exec_args]"`` (NestedMPILauncher),
                         ``"srun [exec_args]"`` (SlurmLauncher, default exec_prefix),
                         ``""`` (ForkLauncher, no wrapper).
    """
    prefix = os.environ.get("COMMAND_PREFIX", "")
    command = f"{prefix} {binary}".strip()
    if ase_env: os.environ[ase_env] = command
    return command


# Directory containing this file — use for artifact paths so they resolve
# correctly regardless of which subdirectory the calculation runs in.
evaluator_dir = Path(__file__).resolve().parent


def evaluator(structure):
    pwx_cmd = build_command("pw.x")
    pseudo_dir = evaluator_dir

    input_data = {
        'control': {
            'calculation': 'scf',
            'tprnfor': True,
            'tstress': True,
        },
        'system': {
            'ecutwfc': 60,
            'ecutrho': 480,
            'nosym': True,
            'occupations': 'smearing',
            'smearing': 'gaussian',
            'degauss': 0.005,
        },
        'electrons': {
            'mixing_beta': 0.7,
            'mixing_mode': 'local-TF',
            'mixing_ndim': 16,
            'diago_david_ndim': 4,
            'conv_thr': 1e-8,
        }
    }

    pseudopotentials = {'Ni': 'ni_pbe_v1.4.uspp.F.UPF', 'Si': 'Si.pbe-n-rrkjus_psl.1.0.0.UPF', 'H': 'H.pbe-rrkjus_psl.1.0.0.UPF'}

    profile = EspressoProfile(command=pwx_cmd, pseudo_dir=pseudo_dir)

    def espresso_calc():
        return ase.calculators.espresso.Espresso(profile=profile, kpts=None, pseudopotentials=pseudopotentials, input_data=input_data)

    structure.calc = espresso_calc()
    structure.get_potential_energy()
    structure.get_forces()
    structure.get_stress()

    return structure


if __name__ == "__main__":
    """Batch-job entry point invoked by SlurmLauncher.call_evaluator().

    SlurmLauncher cannot call evaluator() in-process because structure
    evaluations run on remote compute nodes. Instead it serialises the
    structure to an extxyz file, submits this script via
    ``sbatch --wait --wrap="python evaluator.py <input> <output>"``, and
    reads the result back from the output file once the job finishes.

    COMMAND_PREFIX is inherited from the parent job's environment and
    consumed by build_command() inside evaluator().

    Positional arguments (paths set by SlurmLauncher.call_evaluator):
        input   extxyz file containing the structure to evaluate.
        output  extxyz file where the evaluated structure is written.
    """
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output")
    a = p.parse_args()
    with open(a.input) as f:
        structure = next(ase.io.extxyz.read_extxyz(f))
    ase.io.extxyz.write_extxyz(a.output, [evaluator(structure)])
