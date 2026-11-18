/* -*- c++ -*- ----------------------------------------------------------
   mlip-4's BFGS with its line search.
   Algorithm of mlip-4/src/training/bfgs.{h,cpp} at 79a39bf (Bfgs, Linesearch).
   BSD 2-Clause, see mlip-4's LICENSE. mlip-3's, which MTPFit steps, is
   BFGS in bfgs.h.
------------------------------------------------------------------------- */

#pragma once

#include <string>
#include <vector>

// Minimises f along a ray from 0, given f'(0) < 0, until the Wolfe conditions hold. The
// caller evaluates f and f' at x() and passes them to iterate(), which moves x().
class WolfeLineSearch {
  public:
    WolfeLineSearch() { reset(); }

    // Restarts from 0 with a first step of initial_step.
    void reset(double initial_step = 1.0);
    double x() const { return curr_x; }
    // Moves x() back towards the previous point by ratio.
    void reduce_step(double ratio) { curr_x = prev_x + ratio * (curr_x - prev_x); }
    // Returns false when the search cannot go on.
    bool iterate(double curr_f, double curr_g);

    double wolfe_c1 = 0.1;
    double wolfe_c2 = 0.5;
    std::string message;    // why the last search failed, for the fit's log

  private:
    double curr_x;
    double left_x, left_f, left_g;       // left bound of the minimum
    double right_x, right_f, right_g;    // right bound; it can expand while right_g < 0
    double prev_x, prev_f, prev_g;
};

// The caller evaluates f and its gradient at x() and passes them to iterate(),
// which moves x(). Outside a line search, iterate() also updates the inverse
// Hessian; make_nonlinesearch_iteration() takes a new direction.
class WolfeBFGS {
  public:
    // Sets x and quits any line search. The inverse Hessian is kept unless n changes.
    void set_x(const double* x, int n);
    // Sets the inverse Hessian to the identity.
    void set_identity_as_inv_hess();
    // Zeroes the inverse Hessian's off-diagonal entries.
    void diag_inv_hess();
    void set_wolfe_c1(double value) { wolfe_c1 = linesearch.wolfe_c1 = value; }
    void set_wolfe_c2(double value) { wolfe_c2 = linesearch.wolfe_c2 = value; }

    const std::vector<double>& x() const { return x_; }
    bool in_linesearch() const { return in_linesearch_; }
    // The step length of the last accepted step along its direction, 0 for a stalled search's return to its start.
    double alpha() const { return alpha_; }
    // What the last accepted step set the inverse Hessian to: "diag", "mean" or "" for its update.
    const char* inv_hess_set() const { return inv_hess_set_; }
    // Why iterate or make_nonlinesearch_iteration last returned false.
    const std::string& message() const { return message_; }

    // Returns false when the loss increases along the direction.
    bool iterate(double f, const std::vector<double>& g);
    // Starts a line search along -H g; returns false when that direction ascends.
    bool make_nonlinesearch_iteration(double f, const std::vector<double>& g);

    int max_iters_no_improve = 10;

  private:
    void set_start(double f, const std::vector<double>& g);
    void update_inv_hess(const std::vector<double>& g);
    // Sets the inverse Hessian to the identity times its mean diagonal entry.
    void reset_inv_hess();

    int size = 0;
    std::vector<double> inv_hess;    // [size * size]
    std::vector<double> p;           // search direction
    double p_dot_g = 0.0;
    std::vector<double> x_;
    std::vector<double> x_start, g_start;    // start of the line search
    double f_start = 0.0, p_dot_g_start = 0.0;
    std::vector<double> delta_grad, yC;
    bool in_linesearch_ = false;
    double alpha_ = 0.0;
    const char* inv_hess_set_ = "";
    std::string message_;
    int iters_no_improve = 0;
    int hess_just_diaged = 0;
    WolfeLineSearch linesearch;

    double wolfe_c1 = 0.1;
    double wolfe_c2 = 0.5;
};
