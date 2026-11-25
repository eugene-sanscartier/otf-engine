import numpy
from numpy import int32
import ase
import collections
from typing import IO

from ._core.neighbors import species_indices


def read_cfg(fileobj: IO[str], species: list[str] | dict[int, str] | dict[str, int] | None = None):
    """Read the CFG structures of *fileobj*.

    Each structure holds the 0-based species index of each atom in its ``type_index`` array.

    Args:
        species: Element of each species index: a list (``["Al", "Cu"]``), a dict
            (``{0: "Al", 1: "Cu"}``) or its inverse (``{"Al": 0, "Cu": 1}``).
            Without it, each atom gets the atomic number ``type_index + 1``.
    """
    if isinstance(species, dict) and species and isinstance(next(iter(species)), str):
        species = {v: k for k, v in species.items()}

    lines = collections.deque(fileobj.readlines())
    structures = []
    while lines:
        line = lines.popleft()
        if line.strip() == "BEGIN_CFG":
            size = 0
            energy = None
            supercell, id, type, cartes, f, nbh_grades, stress = [], [], [], [], [], [], []
            calcs, features = {}, {}
            stress_type = None
        elif line.strip() == "Size":
            size = int(lines.popleft().strip())
        elif line.strip() == "Supercell":
            for _ in range(3):
                supercell += [[float(x) for x in lines.popleft().split()]]
        elif line.strip().startswith("AtomData:"):
            fields = line.split()[1:]
            id_i = fields.index("id") if "id" in fields else None
            type_i = fields.index("type") if "type" in fields else None
            cx_i = fields.index("cartes_x") if "cartes_x" in fields else None
            cy_i = fields.index("cartes_y") if "cartes_y" in fields else None
            cz_i = fields.index("cartes_z") if "cartes_z" in fields else None
            fx_i = fields.index("fx") if "fx" in fields else None
            fy_i = fields.index("fy") if "fy" in fields else None
            fz_i = fields.index("fz") if "fz" in fields else None
            nb_i = fields.index("nbh_grades") if "nbh_grades" in fields else None
            for _ in range(size):
                p = lines.popleft().split()
                if id_i is not None:
                    id += [int(p[id_i])]
                if type_i is not None:
                    type += [int(p[type_i])]
                if cx_i is not None and cy_i is not None and cz_i is not None:
                    cartes += [[float(p[cx_i]), float(p[cy_i]), float(p[cz_i])]]
                if fx_i is not None and fy_i is not None and fz_i is not None:
                    f += [[float(p[fx_i]), float(p[fy_i]), float(p[fz_i])]]
                if nb_i is not None:
                    nbh_grades += [float(p[nb_i])]
        elif line.strip() == "Energy":
            energy = float(lines.popleft().strip())
        elif line.strip().startswith("PlusStress:"):
            stress_type = "PlusStress"
            stress = [float(x) for x in lines.popleft().strip().split()]
        elif line.strip().startswith("Feature"):
            _, feature_name, feature_value = line.strip().split()
            features[feature_name] = feature_value
        elif line.strip() == "END_CFG":
            if energy is not None:
                calcs["energy"] = energy
            if f:
                calcs["forces"] = f
            if stress:
                if stress_type == "PlusStress":
                    stress = numpy.array(stress, dtype=float) / abs(numpy.linalg.det(supercell)) * -1
                calcs["stress"] = stress

            if type and species is not None:
                if len(set(type)) > len(species): raise ValueError(f"CFG contains {len(set(type))} distinct species but species={species!r} only defines {len(species)}.")
                atoms = ase.Atoms(symbols=[species[t] for t in type], positions=cartes, cell=supercell, pbc=True)
            else:
                atoms = ase.Atoms(numbers=[t + 1 for t in type], positions=cartes, cell=supercell, pbc=True)
            atoms.calc = ase.calculators.singlepoint.SinglePointCalculator(atoms, **calcs)

            if type:
                atoms.arrays["type_index"] = numpy.array(type, dtype=int32)
            if nbh_grades:
                atoms.set_array("nbh_grades", numpy.array(nbh_grades))
                features["MV_grade"] = numpy.max(nbh_grades)
            if features:
                atoms.info["features"] = features

            structures += [atoms]

    return structures


def write_cfg(fileobj: IO[str], structures: list[ase.Atoms]):
    output = []

    for atoms in structures:
        output += ["BEGIN_CFG"]
        output += [" Size"]
        output += [f"{len(atoms):9d}"]
        output += [" Supercell"]
        cell = atoms.get_cell()
        for row in cell:
            output += [f"    {row[0]:12.6f} {row[1]:12.6f} {row[2]:12.6f}"]

        has_forces = atoms.calc is not None and "forces" in atoms.calc.results
        has_nbh = "nbh_grades" in atoms.arrays

        fields = ["id", "type", "cartes_x", "cartes_y", "cartes_z"]
        if has_forces:
            fields += ["fx", "fy", "fz"]
        if has_nbh:
            fields += ["nbh_grades"]

        type_ranks = species_indices(atoms).tolist()
        positions = atoms.get_positions()
        forces = atoms.calc.results["forces"] if has_forces else None
        nbh = atoms.arrays["nbh_grades"] if has_nbh else None

        output += [" AtomData:  " + "    ".join(fields)]
        for i in range(len(atoms)):
            line = f" {i + 1:9d} {type_ranks[i]:3d} {positions[i][0]:12.6f} {positions[i][1]:12.6f} {positions[i][2]:12.6f} "
            if has_forces:
                line += f" {forces[i][0]:12.6f} {forces[i][1]:12.6f} {forces[i][2]:12.6f} "
            if has_nbh:
                line += f" {nbh[i]:12.6f}"
            output += [line]

        if atoms.calc is not None and "energy" in atoms.calc.results:
            output += [" Energy", f"    {atoms.calc.results['energy']:12.6f}"]

        if atoms.calc is not None and "stress" in atoms.calc.results:
            # PlusStress is -stress * volume, in the ASE Voigt order xx yy zz yz xz xy
            plus_stress = -numpy.asarray(atoms.calc.results["stress"]) * atoms.get_volume()
            output += [" PlusStress:  " + "   ".join(["xx", "yy", "zz", "yz", "xz", "xy"])]
            output += ["    " + " ".join(f"{x:12.6f}" for x in plus_stress)]

        # MV_grade is the largest neighborhood grade
        features = dict(atoms.info.get("features", {}))
        if has_nbh: features["MV_grade"] = numpy.max(nbh)
        for feature_name, feature_value in features.items():
            output += [f" Feature    {feature_name} {feature_value}"]

        output += ["END_CFG"]
        output += [""]

    fileobj.write("\n".join(output) + "\n")
