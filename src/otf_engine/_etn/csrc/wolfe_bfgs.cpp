/* -*- c++ -*- ----------------------------------------------------------
   mlip-4's BFGS with its line search.
   Algorithm of mlip-4/src/training/bfgs.{h,cpp} at 79a39bf (Bfgs, Linesearch).
   BSD 2-Clause, see mlip-4's LICENSE.
------------------------------------------------------------------------- */

#include "wolfe_bfgs.h"
#include "fit_log.h"

#include <cmath>

static const double HUGE_DOUBLE = 1.0e300;

static double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double res = 0.0;
    for (size_t i = 0; i < a.size(); i++)
        res += a[i] * b[i];
    return res;
}

/* ---------------------------------------------------------------------- */
void WolfeLineSearch::reset(double initial_step) {
    left_x = curr_x = 0.0;
    right_x = initial_step;
    right_f = HUGE_DOUBLE;
    right_g = HUGE_DOUBLE;
}

bool WolfeLineSearch::iterate(double curr_f, double curr_g) {
    if (curr_x == 0) {
        // the first iteration
        left_x = prev_x = curr_x;
        left_f = prev_f = curr_f;
        left_g = prev_g = curr_g;
        curr_x = right_x;
        if (curr_g > 0) {
            message = format_text("f'(0) = %.2e > 0", curr_g);
            return false;
        }
        return true;
    }

    if (prev_x == curr_x) {
        message = format_text("prev_x == curr_x = %g", curr_x);
        return false;
    }
    if (curr_x <= 1e-30) {
        message = format_text("curr_x = %.1e <= 1e-30", curr_x);
        return false;
    }

    if (curr_x == right_x || (curr_x > left_x && right_f == HUGE_DOUBLE)) {
        right_x = curr_x;
        right_f = curr_f;
        right_g = curr_g;
    }

    if (right_g < 0 && right_f < left_f) {
        if (curr_x > right_x) {
            left_x = right_x; left_f = right_f; left_g = right_g;
            right_x = curr_x; right_f = curr_f; right_g = curr_g;
        }
    } else {
        if (curr_g < 0 && curr_f < left_f + wolfe_c1 * (curr_x - left_x) * left_g && curr_x > left_x) {
            if (curr_x > right_x) {
                if (curr_f > right_f) {
                    prev_x = curr_x; prev_f = curr_f; prev_g = curr_g;
                    left_x = right_x; left_f = right_f; left_g = right_g;
                    right_x = curr_x; right_f = curr_f; right_g = curr_g;
                    curr_x = 0.5 * (left_x + right_x);
                    return true;
                }
                right_x = curr_x; right_f = curr_f; right_g = curr_g;
            } else {
                left_x = curr_x; left_f = curr_f; left_g = curr_g;
            }
        }

        if (curr_g > 0 && curr_x < right_x) {
            right_x = curr_x;
            right_f = curr_f;
            right_g = curr_g;
        }
    }

    // The sufficient decrease fails: step to the minimum of a logarithmic model
    if (curr_f > left_f + wolfe_c1 * left_g * (curr_x - left_x) + 1e-6 + 1e-7 * curr_f) {
        double slope = -left_g * (curr_x - left_x);
        double f0 = std::log(slope);
        double g0 = left_g / slope;
        double f1 = std::log(curr_f - left_f + slope);

        double next_x = left_x - 0.5 * g0 * (curr_x - left_x) * (curr_x - left_x) / (f1 - f0 - g0 * (curr_x - left_x));

        if (next_x > right_x && right_g > 0)
            next_x = left_x + 0.5 * (curr_x - left_x);
        else if ((next_x - left_x > 3.0 * right_x - 2.0 * left_x) && right_g < 0)
            next_x = left_x + 3.0 * (right_x - left_x);

        prev_x = curr_x; prev_f = curr_f; prev_g = curr_g;
        curr_x = next_x;
        return true;
    }

    if ((curr_g - prev_g) / (curr_x - prev_x) < 0.0) {
        if (right_g < 0) {
            prev_x = curr_x; prev_f = curr_f; prev_g = curr_g;
            curr_x = 3 * right_x - 2 * left_x;
            return true;
        }
        prev_x = curr_x; prev_f = curr_f; prev_g = curr_g;
        curr_x += 0.5 * (right_x - curr_x);
        return true;
    }

    double new_x = curr_x - curr_g * (curr_x - prev_x) / (curr_g - prev_g);
    prev_x = curr_x; prev_f = curr_f; prev_g = curr_g;
    curr_x = new_x;
    if (curr_x > right_x) {
        if (right_g > 0)
            curr_x = 0.5 * left_x + 0.5 * right_x;
        else if (curr_x > 3 * right_x - 2 * left_x)
            curr_x = 3 * right_x - 2 * left_x;
        return true;
    }
    if (curr_x < left_x) {
        prev_x = curr_x; prev_f = curr_f; prev_g = curr_g;
        curr_x = 0.5 * left_x + 0.5 * right_x;
        return true;
    }
    return true;
}

/* ---------------------------------------------------------------------- */
void WolfeBFGS::set_x(const double* x, int n) {
    if (size != n) {
        size = n;
        x_.resize(n);
        x_start.resize(n);
        g_start.resize(n);
        delta_grad.resize(n);
        yC.resize(n);
        p.resize(n);
        set_identity_as_inv_hess();
    }
    std::copy(x, x + n, x_.begin());
    f_start = HUGE_DOUBLE;
    p_dot_g_start = HUGE_DOUBLE;
    linesearch.reset();
    in_linesearch_ = false;
    hess_just_diaged = 0;
}

void WolfeBFGS::set_start(double f, const std::vector<double>& g) {
    x_start = x_;
    g_start = g;
    f_start = f;
    p_dot_g_start = p_dot_g;
}

bool WolfeBFGS::iterate(double f, const std::vector<double>& g) {
    p_dot_g = dot(g, p);

    bool loss_decrease;    // false when the loss increases, which ends BFGS
    if (iters_no_improve < max_iters_no_improve && (f > f_start + wolfe_c1 * linesearch.x() * p_dot_g_start || std::fabs(p_dot_g) > wolfe_c2 * std::fabs(p_dot_g_start))) {
        if (f < 2 * f_start) iters_no_improve++;
        in_linesearch_ = true;

        loss_decrease = linesearch.iterate(f, p_dot_g);
        if (!loss_decrease) {
            iters_no_improve = max_iters_no_improve;
            loss_decrease = true;
        }
        if (iters_no_improve < max_iters_no_improve) {
            for (int i = 0; i < size; i++)
                x_[i] = x_start[i] + linesearch.x() * p[i];
        } else {
            x_ = x_start;
            linesearch.reset();
        }
    } else {
        in_linesearch_ = !in_linesearch_;

        if (!in_linesearch_) {
            if (iters_no_improve >= max_iters_no_improve) {
                // A stalled search restarts from its start with a simpler inverse Hessian
                alpha_ = 0.0;
                if (hess_just_diaged) {
                    reset_inv_hess();
                    inv_hess_set_ = "mean";
                } else {
                    diag_inv_hess();
                    inv_hess_set_ = "diag";
                }
                linesearch.reset();
                x_ = x_start;
                if (hess_just_diaged >= 2) {
                    message_ = format_text("hess_just_diaged %d", hess_just_diaged);
                    return false;
                }
                hess_just_diaged++;
            } else {
                alpha_ = linesearch.x();
                inv_hess_set_ = "";
                update_inv_hess(g);
                hess_just_diaged = 0;
            }
        }

        if (!in_linesearch_) return true;

        loss_decrease = make_nonlinesearch_iteration(f, g);
    }

    return loss_decrease;
}

bool WolfeBFGS::make_nonlinesearch_iteration(double f, const std::vector<double>& g) {
    std::fill(p.begin(), p.end(), 0.0);
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            p[i] -= inv_hess[(size_t) i * size + j] * g[j];
    p_dot_g = dot(g, p);
    iters_no_improve = 0;
    linesearch.reset();
    set_start(f, g);

    if (p_dot_g > 0) {
        message_ = format_text("p.g = %.2e > 0", p_dot_g);
        return false;
    }

    bool loss_decrease = linesearch.iterate(f, p_dot_g);
    if (!loss_decrease) message_ = linesearch.message;
    for (int i = 0; i < size; i++)
        x_[i] = x_start[i] + linesearch.x() * p[i];

    in_linesearch_ = true;
    return loss_decrease;
}

void WolfeBFGS::scale_inv_hess(double c) {
    for (double& h : inv_hess) h *= c;
}

void WolfeBFGS::diag_inv_hess() {
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            if (i != j) inv_hess[(size_t) i * size + j] = 0.0;
}

void WolfeBFGS::set_identity_as_inv_hess() {
    inv_hess.assign((size_t) size * size, 0.0);
    for (int i = 0; i < size; i++)
        inv_hess[(size_t) i * size + i] = 1.0;
}

void WolfeBFGS::reset_inv_hess() {
    double av_diag = 0.0;
    for (int i = 0; i < size; i++)
        av_diag += inv_hess[(size_t) i * size + i];
    av_diag /= size;
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            inv_hess[(size_t) i * size + j] = i != j ? 0.0 : av_diag;
}

void WolfeBFGS::update_inv_hess(const std::vector<double>& g) {
    for (int i = 0; i < size; i++)
        delta_grad[i] = g[i] - g_start[i];

    const double alpha = linesearch.x();
    double py = 0.0;
    double yCy = 0.0;
    for (int i = 0; i < size; i++)
        py += alpha * p[i] * delta_grad[i];

    if (py == 0) return;

    std::fill(yC.begin(), yC.end(), 0.0);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++)
            yC[i] += inv_hess[(size_t) i * size + j] * delta_grad[j];
        yCy += yC[i] * delta_grad[i];
    }

    double foo = (py + yCy) / (py * py);
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            inv_hess[(size_t) i * size + j] += alpha * alpha * p[i] * p[j] * foo - alpha * (p[i] * yC[j] + yC[i] * p[j]) / py;
}
