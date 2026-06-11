import os
import argparse
import ase.io
import ase.io.extxyz
import ase.calculators.espresso
from ase.calculators.espresso import EspressoProfile


def build_command(binary, ase_env=None):
    """Assemble the full DFT executable command from env vars set by the otf-engine launcher.

    The launcher writes ``COMMAND_PREFIX`` before calling evaluator() (or before
    submitting this script as a batch job via SlurmLauncher):

        COMMAND_PREFIX — prefix produced by launcher.command_prefix():
                         ``"mpirun [runner_args]"`` (NestedLauncher),
                         ``"srun [runner_args]"`` (SlurmLauncher, default runner_exec),
                         ``""`` (ForkLauncher, no wrapper).

    Two usage patterns:

    1. Binary declared here:
         launcher = NestedLauncher(runner_args="--bind-to core")
         command = build_command("pw.x")        # → "mpirun --bind-to core pw.x"

    2. ASE env var pattern — lets EspressoProfile read the command automatically:
         build_command("pw.x", ase_env="ASE_ESPRESSO_COMMAND")
         profile = EspressoProfile(pseudo_dir=...)  # no command= needed
    """
    prefix = os.environ.get("COMMAND_PREFIX", "")
    command = f"{prefix} {binary}".strip()
    if ase_env: os.environ[ase_env] = command
    return command


# Directory containing this file — use for artifact paths (pseudopotentials, etc.)
# so they resolve correctly regardless of which subdirectory the calculation runs in.
evaluator_dir = os.path.dirname(__file__) or "."


def evaluator(structure):
    lammps2atomic_numbers = [{1: 28, 2: 14, 3: 1}[z] for z in structure.get_atomic_numbers()]  # cfg reader map by default atomic numbers to atomic type indexes + 1 in order of apearance.
    structure.set_atomic_numbers(lammps2atomic_numbers)

    input_data = {
        'control': {
            'restart_mode': 'restart',
            'calculation': 'scf',
            'etot_conv_thr': 1e-10,
            'forc_conv_thr': 1e-7,
            'tprnfor': True,
            'tstress': True,
        },
        'system': {
            'ecutwfc': 50,
            'ecutrho': 400,
            'nosym': True,
            'occupations': 'smearing',
            'smearing': 'gaussian',
            'degauss': 0.005,
            'starting_magnetization(1)': 0.0,
            'starting_magnetization(2)': 0.7,
        },
    }

    pseudopotentials = {'Ni': 'ni_pbe_v1.4.uspp.F.UPF', 'Si': 'Si.pbe-n-rrkjus_psl.1.0.0.UPF', 'H': 'H_ONCV_PBE-1.0.oncvpsp.upf'}
    command = build_command("pw.x")
    profile = EspressoProfile(command=command, pseudo_dir=evaluator_dir)

    def espresso_calc():
        return ase.calculators.espresso.Espresso(profile=profile, pseudopotentials=pseudopotentials, kpts=None, input_data=input_data)

    structure.calc = espresso_calc()
    structure.get_potential_energy()
    structure.get_forces()
    structure.get_stress()

    return structure


if __name__ == "__main__":
    """Batch-job entry point invoked by SlurmLauncher.call_evaluator().

    SlurmLauncher cannot call evaluator() in-process because structure
    evaluations run on remote compute nodes.  Instead it serialises the
    structure to an extxyz file, submits this script via
    ``sbatch --wait --wrap="python evaluator.py <input> <output>"``, and
    reads the result back from the output file once the job finishes.

    COMMAND_PREFIX is inherited from the parent
    job's environment and consumed by build_command() inside evaluator().

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
