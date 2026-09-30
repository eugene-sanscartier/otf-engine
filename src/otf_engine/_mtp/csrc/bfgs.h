/* -*- c++ -*- ----------------------------------------------------------
   Shapeev's BFGS with its line search, as mlip-3 minimises the training loss.
   Algorithm of mlip-3/src/common/bfgs.{h,cpp}.
   Copyright (c) 2023, Alexander Shapeev (Skoltech). BSD 2-Clause, see LICENSE.mlip-3.
------------------------------------------------------------------------- */

#pragma once

#include <vector>

// Minimises f along a ray from 0, given f'(0) < 0. The caller evaluates f and
// f' at x() and passes them to iterate(), which moves x().
class LineSearch {
  public:
    LineSearch() { reset(); }

    // Restarts from 0 with a first step of initial_step.
    void reset(double initial_step = 1.0);
    double x() const { return curr_x; }
    // Moves x() back towards the previous point by ratio.
    void reduce_step(double ratio);
    // Returns false when the function increases along the ray.
    bool iterate(double curr_f, double curr_g);

  private:
    double curr_x;
    double left_x, left_f, left_g;       // left bound of the minimum
    double right_x, right_f, right_g;    // right bound; it can expand while right_g < 0
    double prev_x, prev_f, prev_g;
};

// The caller evaluates f and its gradient at x() and passes them to iterate(),
// which moves x(). Outside a line search, iterate() also updates the inverse
// Hessian and takes a new direction.
class BFGS {
  public:
    // Sets x and quits any line search. The inverse Hessian is kept unless n changes.
    void set_x(const double* x, int n);
    // Sets the inverse Hessian to the identity.
    void reset_hessian();

    double x(int i) const { return x_[i]; }
    const std::vector<double>& x() const { return x_; }
    bool in_linesearch() const { return in_linesearch_; }

    // Returns false when the loss increases along the direction.
    bool iterate(double f, const std::vector<double>& g);
    void reduce_step(double ratio = 0.25);

  private:
    void update_inv_hess(const std::vector<double>& g);

    int size = 0;
    std::vector<double> inv_hess;    // [size * size]
    std::vector<double> p;           // search direction
    std::vector<double> x_;
    std::vector<double> x_start, g_start;    // start of the line search
    double f_start = 0.0, p_dot_g_start = 0.0;
    std::vector<double> delta_grad, yC;
    bool in_linesearch_ = false;
    LineSearch linesearch;

    const double wolfe_c1 = 0.1;
    const double wolfe_c2 = 0.5;
};
