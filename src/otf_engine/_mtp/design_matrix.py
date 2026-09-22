"""Design matrix assembly for MTP linear fitting.

The unknown vector is ``x = [linear_coeffs | species_coeffs]``. Energy, force
and stress rows are all linear in it, so one ``eval_grad_linear`` pass per
sample supplies every column: the site-energy gradient is the basis matrix, the
force gradient is the force block, and the virial gradient is the stress block.
"""

import numpy
from numpy import float64


def linear_columns(pot, sample, need_forces, need_stress):
    """Return (basis, A_F, A_S) for one sample.

    basis : (n_atoms, n_basis)          d(site energy)/d(linear coeffs)
    A_F   : (n_atoms, 3, n_basis)       d(forces)/d(linear coeffs), or None
    A_S   : (6, n_basis)                d(stress * volume)/d(linear coeffs) in
                                        ASE Voigt order, or None
    """
    if not need_forces and not need_stress:
        return numpy.asarray(pot.eval_basis(sample.neighbors), dtype=float64), None, None

    basis, force_grad, virial_grad = pot.eval_grad_linear(sample.neighbors, need_stress)

    A_S = None
    if need_stress:
        # virial_grad is (xx,yy,zz,xy,xz,yz); stress = -virial, ASE Voigt order.
        vg = numpy.asarray(virial_grad, dtype=float64)
        A_S = -numpy.stack([vg[0], vg[1], vg[2], vg[5], vg[4], vg[3]])

    return numpy.asarray(basis, dtype=float64), numpy.asarray(force_grad, dtype=float64), A_S


def build_design_matrix(pot, dataset, weight_energy=1.0, weight_forces=0.01, weight_stress=0.001, weight_scaling=1, include_forces=True, include_stress=True):
    """Assemble the weighted design matrix and right-hand side.

    Parameters
    ----------
    pot : MTPTraining
    dataset : list of Sample
    weight_energy / weight_forces / weight_stress : float
        Base weights for the three observable types.
    weight_scaling : int
        Exponent for per-config size normalisation: energy and stress rows are
        divided by N^weight_scaling, matching mlip-3's
        wgt_scale_power_energy/stress = 1.  Forces are never scaled
        (mlip-3 wgt_scale_power_forces default = 0).
    include_forces / include_stress : bool
        Whether to add force / stress rows.

    Returns
    -------
    A : ndarray (n_rows, n_basis + species_count)
    b : ndarray (n_rows,)
    """
    n_basis = pot.get_alpha_scalar_count()
    n_species = pot.get_species_count()
    n_params = n_basis + n_species

    A_blocks, b_blocks = [], []

    for sample in dataset:
        n_atoms = sample.n_atoms
        need_forces = include_forces and sample.forces is not None
        need_stress = include_stress and sample.stress is not None

        basis, A_F, A_S = linear_columns(pot, sample, need_forces, need_stress)
        scale = n_atoms**weight_scaling

        # --- Energy row ---
        w_e = weight_energy / scale
        energy_row = numpy.zeros(n_params)
        energy_row[:n_basis] = basis.sum(axis=0)
        types = numpy.asarray(sample.neighbors.types)
        energy_row[n_basis:] = numpy.bincount(types, minlength=n_species)
        A_blocks += [w_e * energy_row[None, :]]
        b_blocks += [numpy.array([w_e * sample.energy])]

        # --- Force rows ---  no per-size scaling
        if need_forces:
            B_F = numpy.zeros((n_atoms * 3, n_params))
            B_F[:, :n_basis] = A_F.reshape(n_atoms * 3, n_basis)
            A_blocks += [weight_forces * B_F]
            b_blocks += [weight_forces * sample.forces.ravel()]

        # --- Stress rows ---
        if need_stress:
            if sample.volume is None:
                raise ValueError("Sample must carry a volume when stress rows are included")
            w_s = weight_stress / scale
            B_S = numpy.zeros((6, n_params))
            B_S[:, :n_basis] = A_S / sample.volume
            A_blocks += [w_s * B_S]
            b_blocks += [w_s * sample.stress]

    A = numpy.vstack(A_blocks)
    b = numpy.concatenate(b_blocks)
    return A, b
