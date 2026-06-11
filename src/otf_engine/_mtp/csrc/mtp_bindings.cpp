/* -*- c++ -*- ----------------------------------------------------------
   pybind11 bindings for MTPPotential.
   Exposes compute(), eval_basis(), eval_radial_basis(), and all parameter
   accessors as numpy-friendly Python objects.
------------------------------------------------------------------------- */

#include "mtp_potential.h"

#include <stdexcept>

#include <Python.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(_mtp_ext, m) {
    m.doc() = "Standalone MTP potential bindings (no LAMMPS).";

    py::class_<MTPPotential>(m, "MTPPotential")
        .def(py::init<const std::string&>(), py::arg("filename"), "Load an MTP potential from a .mtp file (version 1.1.0, RBChebyshev basis).")

        // ----------------------------------------------------------------
        // compute()
        // ----------------------------------------------------------------
        .def(
            "compute",
            [](MTPPotential& self,
               py::array_t<int, py::array::c_style | py::array::forcecast> types,
               py::array_t<int, py::array::c_style | py::array::forcecast> ilist,
               py::array_t<int, py::array::c_style | py::array::forcecast> numneigh,
               py::array_t<int, py::array::c_style | py::array::forcecast> firstneigh,
               py::array_t<double, py::array::c_style | py::array::forcecast> displacements,
               bool compute_virials,
               bool compute_eatom) -> py::dict {
                const int n_atoms = (int)types.shape(0);
                const int inum = (int)ilist.shape(0);

                if (numneigh.shape(0) != inum)
                    throw std::runtime_error("numneigh must have length inum");
                if (displacements.ndim() != 2 || displacements.shape(1) != 3)
                    throw std::runtime_error("displacements must have shape (n_pairs, 3)");

                // Allocate output arrays
                auto forces = py::array_t<double>({n_atoms, 3});
                auto virials = py::array_t<double>({6});
                auto eatom = compute_eatom ? py::array_t<double>({n_atoms})
                                           : py::array_t<double>({0});

                std::fill(forces.mutable_data(), forces.mutable_data() + n_atoms * 3, 0.0);
                std::fill(virials.mutable_data(), virials.mutable_data() + 6, 0.0);
                if (compute_eatom)
                    std::fill(eatom.mutable_data(), eatom.mutable_data() + n_atoms, 0.0);

                double energy = self.compute(
                    n_atoms, types.data(), inum, ilist.data(), numneigh.data(), firstneigh.data(), displacements.data(), forces.mutable_data(), compute_virials ? virials.mutable_data() : nullptr, compute_eatom ? eatom.mutable_data() : nullptr);

                py::dict result;
                result["energy"] = energy;
                result["forces"] = forces;
                result["virials"] = virials;
                if (compute_eatom)
                    result["eatom"] = eatom;
                return result;
            },
            py::arg("types"),
            py::arg("ilist"),
            py::arg("numneigh"),
            py::arg("firstneigh"),
            py::arg("displacements"),
            py::arg("compute_virials") = true,
            py::arg("compute_eatom") = false,
            R"doc(
Compute energy and forces.

Parameters
----------
types         : ndarray int32   (n_atoms)         — 0-indexed MTP species
ilist         : ndarray int32   (inum)             — indices of central atoms
numneigh      : ndarray int32   (inum)             — neighbor count per central atom
firstneigh    : ndarray int32   (sum(numneigh))    — flat original neighbor indices
displacements : ndarray float64 (sum(numneigh), 3) — PBC-corrected r_j - r_i vectors
compute_virials : bool, default True
compute_eatom   : bool, default False

Returns
-------
dict with keys:
  "energy"  : float
  "forces"  : ndarray float64 (n_atoms, 3)
  "virials" : ndarray float64 (6)  — xx,yy,zz,xy,xz,yz
  "eatom"   : ndarray float64 (n_atoms)  — only if compute_eatom=True
)doc")

        // ----------------------------------------------------------------
        // eval_basis()
        // ----------------------------------------------------------------
        .def(
            "eval_basis",
            [](MTPPotential& self,
               py::array_t<int, py::array::c_style | py::array::forcecast> types,
               py::array_t<int, py::array::c_style | py::array::forcecast> ilist,
               py::array_t<int, py::array::c_style | py::array::forcecast> numneigh,
               py::array_t<int, py::array::c_style | py::array::forcecast> firstneigh,
               py::array_t<double, py::array::c_style | py::array::forcecast> displacements)
                -> py::array_t<double> {
                const int n_atoms = (int)types.shape(0);
                const int inum = (int)ilist.shape(0);
                const int n_basis = self.get_alpha_scalar_count();

                auto out = py::array_t<double>({inum, n_basis});
                self.eval_basis(n_atoms, types.data(), inum, ilist.data(), numneigh.data(), firstneigh.data(), displacements.data(), out.mutable_data());
                return out;
            },
            py::arg("types"),
            py::arg("ilist"),
            py::arg("numneigh"),
            py::arg("firstneigh"),
            py::arg("displacements"),
            R"doc(
Evaluate basis function values for each central atom (no linear coefficients applied).

Returns
-------
ndarray float64 (inum, alpha_scalar_count)
  Row ii contains the scalar moment tensor values for atom ilist[ii].
  Dot with get_linear_coeffs() to recover the site energy contribution.
)doc")

        // ----------------------------------------------------------------
        // eval_grad()
        // ----------------------------------------------------------------
        .def(
            "eval_grad",
            [](MTPPotential& self,
               py::array_t<int, py::array::c_style | py::array::forcecast> types,
               py::array_t<int, py::array::c_style | py::array::forcecast> ilist,
               py::array_t<int, py::array::c_style | py::array::forcecast> numneigh,
               py::array_t<int, py::array::c_style | py::array::forcecast> firstneigh,
               py::array_t<double, py::array::c_style | py::array::forcecast> displacements)
                -> py::array_t<double> {
                const int n_atoms = (int)types.shape(0);
                const int inum   = (int)ilist.shape(0);
                const int cc     = self.coeff_count();
                auto out = py::array_t<double>({inum, cc});
                self.eval_grad(n_atoms, types.data(), inum, ilist.data(), numneigh.data(), firstneigh.data(), displacements.data(), out.mutable_data());
                return out;
            },
            py::arg("types"),
            py::arg("ilist"),
            py::arg("numneigh"),
            py::arg("firstneigh"),
            py::arg("displacements"),
            R"doc(
Return the per-atom information vector for extrapolation grade computation.

Returns
-------
ndarray float64 (inum, coeff_count)
  Row ii = gradient of site energy of atom ilist[ii] w.r.t. all MTP coefficients.
  Layout: [radial_grads (radial_coeff_count) | species_one_hot (species_count)
           | basis_values (alpha_scalar_count)]
  coeff_count = radial_coeff_count + species_count + alpha_scalar_count
)doc")

        // ----------------------------------------------------------------
        // eval_radial_basis()
        // ----------------------------------------------------------------
        .def(
            "eval_radial_basis",
            [](MTPPotential& self, double dist) -> py::tuple {
                const int sz = self.get_radial_basis_size();
                auto vals = py::array_t<double>({sz});
                auto ders = py::array_t<double>({sz});
                self.eval_radial_basis(dist, vals.mutable_data(), ders.mutable_data());
                return py::make_tuple(vals, ders);
            },
            py::arg("dist"),
            "Return (vals, ders) of the Chebyshev radial basis at distance dist.")

        // ----------------------------------------------------------------
        // Scalar parameter accessors
        // ----------------------------------------------------------------
        .def("get_coeff_count", &MTPPotential::coeff_count, "radial_coeff_count + species_count + alpha_scalar_count")
        .def("get_species_count", &MTPPotential::get_species_count)
        .def("get_radial_func_count", &MTPPotential::get_radial_func_count)
        .def("get_radial_basis_size", &MTPPotential::get_radial_basis_size)
        .def("get_alpha_scalar_count", &MTPPotential::get_alpha_scalar_count)
        .def("get_alpha_moment_count", &MTPPotential::get_alpha_moment_count)
        .def("get_alpha_index_basic_count", &MTPPotential::get_alpha_index_basic_count)
        .def("get_alpha_index_times_count", &MTPPotential::get_alpha_index_times_count)
        .def("get_min_cutoff", &MTPPotential::get_min_cutoff)
        .def("get_max_cutoff", &MTPPotential::get_max_cutoff)
        .def("set_min_cutoff", &MTPPotential::set_min_cutoff, py::arg("d"),
             "Update radial basis minimum distance. Mirrors mlip-3 AddSpecies(): min_val = 0.99 * min(training distances).")
        .def("get_scaling", &MTPPotential::get_scaling)
        .def("set_scaling", &MTPPotential::set_scaling, py::arg("scaling"), "Set the global scaling parameter.")
        .def("get_radial_coeff_count", &MTPPotential::get_radial_coeff_count, "Total number of radial basis coefficients (species^2 * radial_func_count * radial_basis_size).")
        .def("get_potential_name", &MTPPotential::get_potential_name)

        // ----------------------------------------------------------------
        // Array parameter accessors — return numpy views (zero-copy)
        // ----------------------------------------------------------------
        .def(
            "get_radial_basis_coeffs",
            [](const MTPPotential& self) {
                const int sz = self.get_species_count() * self.get_species_count() *
                               self.get_radial_func_count() * self.get_radial_basis_size();
                return py::array_t<double>(
                    {self.get_species_count() * self.get_species_count(),
                     self.get_radial_func_count(),
                     self.get_radial_basis_size()},
                    self.get_radial_basis_coeffs());
            },
            "ndarray float64 (species^2, radial_func_count, radial_basis_size)")

        .def(
            "get_alpha_index_basic",
            [](const MTPPotential& self) {
                return py::array_t<int>(
                    {self.get_alpha_index_basic_count(), 4},
                    self.get_alpha_index_basic());
            },
            "ndarray int32 (alpha_index_basic_count, 4)  columns: mu, px, py, pz")

        .def(
            "get_alpha_index_times",
            [](const MTPPotential& self) {
                return py::array_t<int>(
                    {self.get_alpha_index_times_count(), 4},
                    self.get_alpha_index_times());
            },
            "ndarray int32 (alpha_index_times_count, 4)  columns: i0, i1, multiplier, out")

        .def(
            "get_alpha_moment_mapping",
            [](const MTPPotential& self) {
                return py::array_t<int>(
                    {self.get_alpha_scalar_count()},
                    self.get_alpha_moment_mapping());
            },
            "ndarray int32 (alpha_scalar_count)")

        .def(
            "get_linear_coeffs",
            [](const MTPPotential& self) {
                return py::array_t<double>(
                    {self.get_alpha_scalar_count()},
                    self.get_linear_coeffs());
            },
            "ndarray float64 (alpha_scalar_count)  — moment tensor basis coefficients")

        .def(
            "get_species_coeffs",
            [](const MTPPotential& self) {
                return py::array_t<double>(
                    {self.get_species_count()},
                    self.get_species_coeffs());
            },
            "ndarray float64 (species_count)  — per-species reference energies")

        // ----------------------------------------------------------------
        // Coefficient setters (for training)
        // ----------------------------------------------------------------
        .def(
            "set_linear_coeffs",
            [](MTPPotential& self, py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
                if (arr.size() != self.get_alpha_scalar_count())
                    throw std::runtime_error(
                        "set_linear_coeffs: expected length " +
                        std::to_string(self.get_alpha_scalar_count()));
                self.set_linear_coeffs(arr.data());
            },
            py::arg("coeffs"),
            "Set linear (moment tensor) coefficients. Shape: (alpha_scalar_count,)")

        .def(
            "set_species_coeffs",
            [](MTPPotential& self, py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
                if (arr.size() != self.get_species_count())
                    throw std::runtime_error(
                        "set_species_coeffs: expected length " +
                        std::to_string(self.get_species_count()));
                self.set_species_coeffs(arr.data());
            },
            py::arg("coeffs"),
            "Set per-species reference energies. Shape: (species_count,)")

        .def(
            "set_radial_basis_coeffs",
            [](MTPPotential& self, py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
                const int expected = self.get_species_count() * self.get_species_count() *
                                     self.get_radial_func_count() * self.get_radial_basis_size();
                if (arr.size() != expected)
                    throw std::runtime_error(
                        "set_radial_basis_coeffs: expected length " + std::to_string(expected));
                self.set_radial_basis_coeffs(arr.data());
            },
            py::arg("coeffs"),
            "Set radial basis coefficients. Shape: (species^2, radial_func_count, radial_basis_size) or flat.")

        // ----------------------------------------------------------------
        // compute_with_radial_grad()
        // ----------------------------------------------------------------
        .def(
            "compute_with_radial_grad",
            [](MTPPotential& self,
               py::array_t<int, py::array::c_style | py::array::forcecast> types,
               py::array_t<int, py::array::c_style | py::array::forcecast> ilist,
               py::array_t<int, py::array::c_style | py::array::forcecast> numneigh,
               py::array_t<int, py::array::c_style | py::array::forcecast> firstneigh,
               py::array_t<double, py::array::c_style | py::array::forcecast> displacements,
               bool compute_virial_grad) -> py::tuple {
                const int n_atoms  = (int)types.shape(0);
                const int inum     = (int)ilist.shape(0);
                const int n_radial = self.get_radial_coeff_count();

                auto forces      = py::array_t<double>({n_atoms, 3});
                auto virials     = py::array_t<double>({6});
                auto energy_grad = py::array_t<double>({n_radial});
                auto force_grad  = py::array_t<double>({n_atoms, 3, n_radial});

                std::fill(forces.mutable_data(),      forces.mutable_data()      + n_atoms * 3,    0.0);
                std::fill(virials.mutable_data(),     virials.mutable_data()     + 6,              0.0);
                std::fill(energy_grad.mutable_data(), energy_grad.mutable_data() + n_radial,       0.0);
                std::fill(force_grad.mutable_data(),  force_grad.mutable_data()  + (long)n_atoms * 3 * n_radial, 0.0);

                py::array_t<double> virial_grad;
                double* vg_ptr = nullptr;
                if (compute_virial_grad) {
                    virial_grad = py::array_t<double>({6, n_radial});
                    std::fill(virial_grad.mutable_data(), virial_grad.mutable_data() + 6 * n_radial, 0.0);
                    vg_ptr = virial_grad.mutable_data();
                }

                double energy = self.compute_with_radial_grad(
                    n_atoms, types.data(), inum, ilist.data(), numneigh.data(),
                    firstneigh.data(), displacements.data(),
                    forces.mutable_data(), virials.mutable_data(),
                    energy_grad.mutable_data(), force_grad.mutable_data(), vg_ptr);

                if (compute_virial_grad)
                    return py::make_tuple(energy, forces, virials, energy_grad, force_grad, virial_grad);
                return py::make_tuple(energy, forces, virials, energy_grad, force_grad, py::none());
            },
            py::arg("types"),
            py::arg("ilist"),
            py::arg("numneigh"),
            py::arg("firstneigh"),
            py::arg("displacements"),
            py::arg("compute_virial_grad") = false,
            R"doc(
Fused compute + energy/force/virial radial gradients in one neighborhood pass.

Returns
-------
(energy, forces, virials, energy_grad, force_grad, virial_grad) where:
  energy      : float
  forces      : ndarray float64 (n_atoms, 3)
  virials     : ndarray float64 (6)  — xx,yy,zz,xy,xz,yz
  energy_grad : ndarray float64 (radial_coeff_count)  — ∂E_total/∂c_r
  force_grad  : ndarray float64 (n_atoms, 3, radial_coeff_count)
  virial_grad : ndarray float64 (6, radial_coeff_count) or None
)doc")

        // ----------------------------------------------------------------
        // eval_grad_radial / eval_grad_linear / eval_grad_all
        // ----------------------------------------------------------------
        .def(
            "eval_grad_radial",
            [](MTPPotential& self,
               py::array_t<int, py::array::c_style | py::array::forcecast> types,
               py::array_t<int, py::array::c_style | py::array::forcecast> ilist,
               py::array_t<int, py::array::c_style | py::array::forcecast> numneigh,
               py::array_t<int, py::array::c_style | py::array::forcecast> firstneigh,
               py::array_t<double, py::array::c_style | py::array::forcecast> displacements,
               bool compute_virial_grad) -> py::tuple {
                const int n_atoms = (int)types.shape(0), inum = (int)ilist.shape(0);
                const int nr = self.get_radial_coeff_count();
                auto eg = py::array_t<double>({nr});
                auto fg = py::array_t<double>({n_atoms, 3, nr});
                py::array_t<double> vg;
                double* vg_ptr = nullptr;
                if (compute_virial_grad) { vg = py::array_t<double>({6, nr}); std::fill(vg.mutable_data(), vg.mutable_data()+6*nr, 0.0); vg_ptr = vg.mutable_data(); }
                std::fill(eg.mutable_data(), eg.mutable_data()+nr, 0.0);
                std::fill(fg.mutable_data(), fg.mutable_data()+(long)n_atoms*3*nr, 0.0);
                self.eval_grad_radial(n_atoms, types.data(), inum, ilist.data(), numneigh.data(), firstneigh.data(), displacements.data(), eg.mutable_data(), fg.mutable_data(), vg_ptr);
                return py::make_tuple(eg, fg, compute_virial_grad ? py::object(vg) : py::none());
            },
            py::arg("types"), py::arg("ilist"), py::arg("numneigh"), py::arg("firstneigh"), py::arg("displacements"), py::arg("compute_virial_grad") = false,
            "Returns (energy_grad(n_radial), force_grad(n,3,n_radial), virial_grad(6,n_radial)|None)")

        .def(
            "eval_grad_linear",
            [](MTPPotential& self,
               py::array_t<int, py::array::c_style | py::array::forcecast> types,
               py::array_t<int, py::array::c_style | py::array::forcecast> ilist,
               py::array_t<int, py::array::c_style | py::array::forcecast> numneigh,
               py::array_t<int, py::array::c_style | py::array::forcecast> firstneigh,
               py::array_t<double, py::array::c_style | py::array::forcecast> displacements,
               bool compute_virial_grad) -> py::tuple {
                const int n_atoms = (int)types.shape(0), inum = (int)ilist.shape(0);
                const int nl = self.get_alpha_scalar_count();
                auto eg = py::array_t<double>({inum, nl});
                auto fg = py::array_t<double>({n_atoms, 3, nl});
                py::array_t<double> vg;
                double* vg_ptr = nullptr;
                if (compute_virial_grad) { vg = py::array_t<double>({6, nl}); std::fill(vg.mutable_data(), vg.mutable_data()+6*nl, 0.0); vg_ptr = vg.mutable_data(); }
                std::fill(eg.mutable_data(), eg.mutable_data()+(long)inum*nl, 0.0);
                std::fill(fg.mutable_data(), fg.mutable_data()+(long)n_atoms*3*nl, 0.0);
                self.eval_grad_linear(n_atoms, types.data(), inum, ilist.data(), numneigh.data(), firstneigh.data(), displacements.data(), eg.mutable_data(), fg.mutable_data(), vg_ptr);
                return py::make_tuple(eg, fg, compute_virial_grad ? py::object(vg) : py::none());
            },
            py::arg("types"), py::arg("ilist"), py::arg("numneigh"), py::arg("firstneigh"), py::arg("displacements"), py::arg("compute_virial_grad") = false,
            "Returns (site_energy_grad(inum,n_lin), force_grad(n,3,n_lin), virial_grad(6,n_lin)|None)")

        .def(
            "eval_grad_all",
            [](MTPPotential& self,
               py::array_t<int, py::array::c_style | py::array::forcecast> types,
               py::array_t<int, py::array::c_style | py::array::forcecast> ilist,
               py::array_t<int, py::array::c_style | py::array::forcecast> numneigh,
               py::array_t<int, py::array::c_style | py::array::forcecast> firstneigh,
               py::array_t<double, py::array::c_style | py::array::forcecast> displacements,
               bool compute_virial_grad) -> py::tuple {
                const int n_atoms = (int)types.shape(0), inum = (int)ilist.shape(0);
                const int cc = self.coeff_count();
                auto eg = py::array_t<double>({inum, cc});
                auto fg = py::array_t<double>({n_atoms, 3, cc});
                py::array_t<double> vg;
                double* vg_ptr = nullptr;
                if (compute_virial_grad) { vg = py::array_t<double>({6, cc}); std::fill(vg.mutable_data(), vg.mutable_data()+6*cc, 0.0); vg_ptr = vg.mutable_data(); }
                std::fill(eg.mutable_data(), eg.mutable_data()+(long)inum*cc, 0.0);
                std::fill(fg.mutable_data(), fg.mutable_data()+(long)n_atoms*3*cc, 0.0);
                self.eval_grad_all(n_atoms, types.data(), inum, ilist.data(), numneigh.data(), firstneigh.data(), displacements.data(), eg.mutable_data(), fg.mutable_data(), vg_ptr);
                return py::make_tuple(eg, fg, compute_virial_grad ? py::object(vg) : py::none());
            },
            py::arg("types"), py::arg("ilist"), py::arg("numneigh"), py::arg("firstneigh"), py::arg("displacements"), py::arg("compute_virial_grad") = false,
            R"doc(
∂(E_i, F_i, virial) / ∂c_all where c_all = [c_radial | c_species | β_linear].

Returns (site_energy_grad(inum, cc), force_grad(n_atoms,3,cc), virial_grad(6,cc)|None)
cc = radial_coeff_count + species_count + alpha_scalar_count
)doc");
}
