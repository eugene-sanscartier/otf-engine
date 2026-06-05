import os
import ase
import ase.calculators
import ase.calculators.lammpsrun
import ase.calculators.espresso

from ase.calculators.espresso import EspressoProfile

def evaluator(structure):
    lammps2atomic_numbers = [{1: 28, 2: 14, 3: 1}[z]
                             for z in structure.get_atomic_numbers()]
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

    pseudopotentials = {
        'Ni': 'ni_pbe_v1.4.uspp.F.UPF',
        'Si': 'Si.pbe-n-rrkjus_psl.1.0.0.UPF',
        'H': 'H_ONCV_PBE-1.0.oncvpsp.upf'
    }

    profile = EspressoProfile(
        command='mpiexec -n 4 pw.x',
        pseudo_dir='.')

    def espresso_calc(): return ase.calculators.espresso.Espresso(profile=profile, pseudopotentials=pseudopotentials, kpts=None, input_data=input_data)
    # def espresso_calc(): return ase.calculators.espresso.Espresso(pseudopotentials=pseudopotentials, kpts=None, input_data=input_data) # kpts=(0, 0, 0)

    structure.calc = espresso_calc()
    structure.get_potential_energy()
    structure.get_forces()
    structure.get_stress()

    return structure
