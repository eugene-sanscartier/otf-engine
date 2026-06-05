import numpy
from numpy import int32
import ase
import collections


def read_cfg(fileobj, type_map=None):
    """Read one or more CFG structures.

    Parameters
    ----------
    fileobj : file-like
    type_map : dict[int, str] or None
        Maps 0-indexed MTP type integers to element symbols
        (e.g. ``{0: "Al", 1: "Cu"}``).  When provided, chemical symbols are
        set from this mapping.  When absent, atomic numbers are set to
        ``type_index + 1`` as a placeholder and the ``type_index`` array on
        each Atoms carries the authoritative 0-indexed type.
    """
    lines = collections.deque(fileobj.readlines())
    images = []
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
                    stress = numpy.array(stress, dtype=float) / numpy.linalg.det(supercell) * -1
                calcs["stress"] = stress

            if type and type_map is not None:
                atoms = ase.Atoms(symbols=[type_map[t] for t in type], positions=cartes, cell=supercell, pbc=True)
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

            images += [atoms]

    return images


def write_cfg(fileobj, images, fmt='%12.6f'):

    def map2ranks(arr):
        seen, types = set(), []
        for x in arr:
            if x not in seen:
                seen.add(x)
                types.append(x)
        rank_map = {val: i for i, val in enumerate(types)}
        return [rank_map[x] for x in arr]

    output = []

    for atoms in images:
        output += ["BEGIN_CFG\n"]
        output += [" Size\n"]
        output += ["%9d\n" % len(atoms)]
        output += [" Supercell\n"]
        cell = atoms.get_cell()
        for row in cell:
            output += ["    {} {} {}\n".format(fmt % row[0], fmt % row[1], fmt % row[2])]

        has_forces = atoms.calc is not None and "forces" in atoms.calc.results
        has_nbh = "nbh_grades" in atoms.arrays

        fields = ["id", "type", "cartes_x", "cartes_y", "cartes_z"]
        if has_forces:
            fields += ["fx", "fy", "fz"]
        if has_nbh:
            fields += ["nbh_grades"]

        if "type_index" in atoms.arrays:
            type_ranks = atoms.arrays["type_index"].tolist()
        else:
            type_ranks = map2ranks(atoms.get_chemical_symbols())
        positions = atoms.get_positions()
        forces = atoms.calc.results["forces"] if has_forces else None
        nbh = atoms.arrays["nbh_grades"] if has_nbh else None

        fields_data = {"id": [], "type": [], "cartes_x": [], "cartes_y": [], "cartes_z": []}
        if has_forces:
            fields_data["fx"], fields_data["fy"], fields_data["fz"] = [], [], []
        if has_nbh:
            fields_data["nbh_grades"] = []

        for i in range(len(atoms)):
            fields_data["id"] += [i + 1]
            fields_data["type"] += [type_ranks[i]]
            fields_data["cartes_x"] += [positions[i][0]]
            fields_data["cartes_y"] += [positions[i][1]]
            fields_data["cartes_z"] += [positions[i][2]]
            if has_forces:
                fields_data["fx"] += [forces[i][0]]
                fields_data["fy"] += [forces[i][1]]
                fields_data["fz"] += [forces[i][2]]
            if has_nbh:
                fields_data["nbh_grades"] += [nbh[i]]

        if "nbh_grades" in atoms.arrays:
            if "features" in atoms.info:
                atoms.info["features"]["MV_grade"] = numpy.max(atoms.arrays["nbh_grades"])
            else:
                atoms.info["features"] = {"MV_grade": numpy.max(atoms.arrays["nbh_grades"])}

        has_energy = atoms.calc is not None and "energy" in atoms.calc.results
        has_stress = atoms.calc is not None and "stress" in atoms.calc.results

        output += [" AtomData:  " + "    ".join(fields) + "\n"]
        for i in range(len(atoms)):
            line = " {:9d} {:3d} {} {} {} ".format(fields_data["id"][i], fields_data["type"][i], fmt % fields_data["cartes_x"][i], fmt % fields_data["cartes_y"][i], fmt % fields_data["cartes_z"][i])
            if has_forces:
                line += " {} {} {} ".format(fmt % fields_data["fx"][i], fmt % fields_data["fy"][i], fmt % fields_data["fz"][i])
            if has_nbh:
                line += " {}".format(fmt % fields_data["nbh_grades"][i])
            line += "\n"
            output += [line]

        output += [" Energy\n"]
        if has_energy:
            output += ["    {}\n".format(fmt % atoms.calc.results["energy"])]
        else:
            output += ["    {}\n".format(fmt % 0.0)]

        if has_stress:
            stress_fields, stress_fields_data = ["xx", "yy", "zz", "yz", "xz", "xy"], {}
            for i, stress_field in enumerate(stress_fields):
                stress_fields_data[stress_field] = atoms.calc.results["stress"][i] * atoms.get_volume() * -1
            output += [" PlusStress:  " + "   ".join(stress_fields) + "\n"]
            output += ["    {} {} {} {} {} {}\n".format(fmt % stress_fields_data["xx"], fmt % stress_fields_data["yy"], fmt % stress_fields_data["zz"], fmt % stress_fields_data["yz"], fmt % stress_fields_data["xz"], fmt % stress_fields_data["xy"])]

        if "features" in atoms.info:
            for feature_name, feature_value in atoms.info["features"].items():
                output += [" Feature    {} {}\n".format(feature_name, feature_value)]

        output += ["END_CFG\n"]
        output += ["\n"]

    fileobj.write("".join(output))
