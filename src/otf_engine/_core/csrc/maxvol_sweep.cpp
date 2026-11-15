/* -*- c++ -*- ----------------------------------------------------------
   MaxVol's sweep (see maxvol_sweep.h). Built with SWEEP_GRADES naming the
   function and Eigen defined to a namespace of this build's own.
------------------------------------------------------------------------- */

#define EIGEN_DONT_PARALLELIZE
#include <Eigen/Core>

#include "maxvol_sweep.h"

void SWEEP_GRADES(const double* rows, int count, const double* invA, int n, double* block, double* grades) {
    using RowMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    const Eigen::Map<const RowMatrix> inverse(invA, n, n);
    for (int start = 0; start < count; start += sweep_block_rows) {
        const int b = count - start < sweep_block_rows ? count - start : sweep_block_rows;
        Eigen::Map<RowMatrix> product(block, b, n);
        product.noalias() = Eigen::Map<const RowMatrix>(rows + (size_t) start * n, b, n) * inverse.transpose();
        Eigen::Map<Eigen::VectorXd>(grades + start, b) = product.cwiseAbs().rowwise().maxCoeff();
    }
}
