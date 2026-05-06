import numpy
import ase
import collections


def read_cfg(fileobj):
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
                if id_i is not None: id += [int(p[id_i])]
                if type_i is not None: type += [int(p[type_i]) + 1]
                if cx_i is not None and cy_i is not None and cz_i is not None: cartes += [[float(p[cx_i]), float(p[cy_i]), float(p[cz_i])]]
                if fx_i is not None and fy_i is not None and fz_i is not None: f += [[float(p[fx_i]), float(p[fy_i]), float(p[fz_i])]]
                if nb_i is not None: nbh_grades += [float(p[nb_i])]
        elif line.strip() == "Energy":
            energy = float(lines.popleft().strip())
        elif line.strip().startswith("PlusStress:"):
            stress_type = "PlusStress"
            stress = [float(x) for x in lines.popleft().strip().split()]
        elif line.strip().startswith("Feature"):
            _, feature_name, feature_value = line.strip().split()
            features[feature_name] = feature_value
        elif line.strip() == "END_CFG":
            if energy != None: calcs["energy"] = energy
            if f != []: calcs["forces"] = f
            if stress != []:
                if stress_type == "PlusStress": stress = numpy.array(stress, dtype=float) / numpy.linalg.det(supercell) * -1
                calcs["stress"] = stress

            atoms = ase.Atoms(numbers=type, positions=cartes, cell=supercell, pbc=True)
            atoms.calc = ase.calculators.singlepoint.SinglePointCalculator(atoms, **calcs)

            if nbh_grades != []:
                atoms.set_array("nbh_grades", numpy.array(nbh_grades))
                features["MV_grade"] = numpy.max(nbh_grades)
            if features != {}: atoms.info["features"] = features

            images += [atoms]

    return images


def write_cfg(fileobj, images, fmt='%12.6f'):

    def map2ranks(arr):
        types = []
        _none = [types.append(type) for type in arr if type not in types]
        rank_map = {val: i for i, val in enumerate(types)}
        return [rank_map[num] for num in arr]

    output = []

    for atoms in images:
        output += ["BEGIN_CFG\n"]
        output += [" Size\n"]
        output += ["%9d\n" % len(atoms)]
        output += [" Supercell\n"]
        output += ["    {} {} {}\n".format(fmt % atoms.get_cell()[0][0], fmt % atoms.get_cell()[0][1], fmt % atoms.get_cell()[0][2])]
        output += ["    {} {} {}\n".format(fmt % atoms.get_cell()[1][0], fmt % atoms.get_cell()[1][1], fmt % atoms.get_cell()[1][2])]
        output += ["    {} {} {}\n".format(fmt % atoms.get_cell()[2][0], fmt % atoms.get_cell()[2][1], fmt % atoms.get_cell()[2][2])]

        fields, fields_data = ["id", "type", "cartes_x", "cartes_y", "cartes_z"], {"id": [], "type": [], "cartes_x": [], "cartes_y": [], "cartes_z": []}
        if atoms.calc != None and "forces" in atoms.calc.results:
            fields += ["fx", "fy", "fz"]
            fields_data["fx"], fields_data["fy"], fields_data["fz"] = [], [], []
        if "nbh_grades" in atoms.arrays:
            fields += ["nbh_grades"]
            fields_data["nbh_grades"] = []

        for i in range(len(atoms)):
            fields_data["id"] += [i + 1]
            fields_data["type"] += [map2ranks(atoms.get_atomic_numbers())[i]]
            fields_data["cartes_x"] += [atoms.get_positions()[i][0]]
            fields_data["cartes_y"] += [atoms.get_positions()[i][1]]
            fields_data["cartes_z"] += [atoms.get_positions()[i][2]]
            if atoms.calc != None and "forces" in atoms.calc.results:
                fields_data["fx"] += [atoms.calc.results["forces"][i][0]]
                fields_data["fy"] += [atoms.calc.results["forces"][i][1]]
                fields_data["fz"] += [atoms.calc.results["forces"][i][2]]
            if "nbh_grades" in atoms.arrays:
                fields_data["nbh_grades"] += [atoms.arrays["nbh_grades"][i]]

        if "nbh_grades" in atoms.arrays:
            if "features" in atoms.info:
                atoms.info["features"]["MV_grade"] = numpy.max(atoms.arrays["nbh_grades"])
            else:
                atoms.info["features"] = {"MV_grade": numpy.max(atoms.arrays["nbh_grades"])}

        output += [" AtomData:  " + "    ".join(fields) + "\n"]
        for i in range(len(atoms)):
            line = " {:9d} {:3d} {} {} {} ".format(fields_data["id"][i], fields_data["type"][i], fmt % fields_data["cartes_x"][i], fmt % fields_data["cartes_y"][i], fmt % fields_data["cartes_z"][i])
            if atoms.calc != None and "forces" in atoms.calc.results:
                line += " {} {} {} ".format(fmt % fields_data["fx"][i], fmt % fields_data["fy"][i], fmt % fields_data["fz"][i])
            if "nbh_grades" in atoms.arrays:
                line += " {}".format(fmt % fields_data["nbh_grades"][i])
            line += "\n"
            output += [line]

        output += [" Energy\n"]
        if atoms.calc != None and "energy" in atoms.calc.results:
            output += ["    {}\n".format(fmt % atoms.calc.results["energy"])]
        else:
            output += ["    {}\n".format(fmt % 0.0)]

        if atoms.calc != None and "stress" in atoms.calc.results:
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
