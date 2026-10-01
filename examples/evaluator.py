"""Evaluator for otf_engine: evaluator(structure) labels one structure with Quantum ESPRESSO.

The engine runs it in each evaluation's directory through ``python -m otf_engine.evaluate``.
"""
from pathlib import Path

import ase.calculators.espresso
from ase.calculators.espresso import EspressoProfile

from otf_engine.evaluate import command


# Directory containing this file — use for artifact paths so they resolve
# correctly regardless of which subdirectory the calculation runs in.
evaluator_dir = Path(__file__).resolve().parent


def evaluator(structure):
    pwx_cmd = command("pw.x")
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
