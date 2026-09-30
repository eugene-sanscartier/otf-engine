/* -*- c++ -*- ----------------------------------------------------------
   Shapeev's BFGS with its line search.
   Algorithm of mlip-3/src/common/bfgs.{h,cpp}.
   Copyright (c) 2023, Alexander Shapeev (Skoltech). BSD 2-Clause, see LICENSE.mlip-3.
------------------------------------------------------------------------- */

#include "bfgs.h"

#include <cmath>
#include <iostream>

static const double HUGE_DOUBLE = 9.9e99;

static double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double res = 0.0;
    for (size_t i = 0; i < a.size(); i++)
        res += a[i] * b[i];
    return res;
}

static void warning(const char* message) {
    std::cerr << "WARNING: " << message << std::endl;
}

/* ---------------------------------------------------------------------- */
void LineSearch::reset(double initial_step) {
    left_x = curr_x = 0.0;
    right_x = initial_step;
    right_f = HUGE_DOUBLE;
    right_g = HUGE_DOUBLE;
}

void LineSearch::reduce_step(double ratio) {
    curr_x = prev_x + ratio * (curr_x - prev_x);
}

bool LineSearch::iterate(double curr_f, double curr_g) {
    if (curr_x == 0) {
        // the first iteration
        left_x = prev_x = curr_x;
        left_f = prev_f = curr_f;
        left_g = prev_g = curr_g;
        curr_x = right_x;
        if (curr_g > 0) {
            warning("Linesearch with increasing funcion!");
            return false;
        }
        return true;
    }

    if (prev_x == curr_x) {
        warning("prev_x == curr_x");
        return false;
    }

    if (curr_x == right_x) {
        right_x = curr_x;
        right_f = curr_f;
        right_g = curr_g;
    }

    // update the bounds
    if (right_g < 0 && right_f < left_f) {
        if (curr_x > right_x) {
            left_x = right_x; left_f = right_f; left_g = right_g;
            right_x = curr_x; right_f = curr_f; right_g = curr_g;
        }
    } else {
        if (curr_g < 0 && curr_f < left_f && curr_x > left_x) {
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
            right_x = curr_x; right_f = curr_f; right_g = curr_g;
        }
    }

    // weak Wolfe conditions
    if (curr_f > left_f + 0.1 * left_g * (curr_x - left_x)) {
        // fit log(f - delta_f) with a parabola through left_f, left_g, curr_f
        const double slope = -left_g * (curr_x - left_x);
        const double f0 = std::log(slope);
        const double g0 = left_g / slope;
        const double f1 = std::log(curr_f - left_f + slope);
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
        // negative second derivative
        prev_x = curr_x; prev_f = curr_f; prev_g = curr_g;
        if (right_g < 0)
            curr_x = 3 * right_x - 2 * left_x;
        else
            curr_x += 0.5 * (right_x - curr_x);
        return true;
    }

    // secant step
    const double new_x = curr_x - curr_g * (curr_x - prev_x) / (curr_g - prev_g);
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
    }
    return true;
}

/* ---------------------------------------------------------------------- */
void BFGS::set_x(const double* x, int n) {
    if (size != n) {
        size = n;
        x_.assign(n, 0.0);
        x_start.assign(n, 0.0);
        g_start.assign(n, 0.0);
        delta_grad.assign(n, 0.0);
        yC.assign(n, 0.0);
        p.assign(n, 0.0);
        reset_hessian();
    }
    std::copy(x, x + n, x_.begin());
    f_start = HUGE_DOUBLE;
    p_dot_g_start = HUGE_DOUBLE;
    linesearch.reset();
    in_linesearch_ = false;
}

void BFGS::reset_hessian() {
    inv_hess.assign((size_t) size * size, 0.0);
    for (int i = 0; i < size; i++)
        inv_hess[(size_t) i * size + i] = 1.0;
}

bool BFGS::iterate(double f, const std::vector<double>& g) {
    double p_dot_g = dot(g, p);
    bool loss_decrease = true;

    if (f > f_start + wolfe_c1 * linesearch.x() * p_dot_g_start || std::fabs(p_dot_g) > wolfe_c2 * std::fabs(p_dot_g_start)) {
        // the loss increased or decreased too little: continue the line search
        in_linesearch_ = true;
        loss_decrease = linesearch.iterate(f, p_dot_g);
    } else {
        // Toggled rather than cleared, so the caller cannot be caught in an infinite cycle.
        in_linesearch_ = !in_linesearch_;

        update_inv_hess(g);

        std::fill(p.begin(), p.end(), 0.0);
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                p[i] -= inv_hess[(size_t) i * size + j] * g[j];
        p_dot_g = dot(g, p);
        linesearch.reset();

        x_start = x_;
        g_start = g;
        f_start = f;
        p_dot_g_start = p_dot_g;

        if (p_dot_g > 0)
            warning("BFGS: stepping in accend direction detected.");

        loss_decrease = linesearch.iterate(f, p_dot_g);
    }

    for (int i = 0; i < size; i++)
        x_[i] = x_start[i] + linesearch.x() * p[i];
    return loss_decrease;
}

void BFGS::reduce_step(double ratio) {
    linesearch.reduce_step(ratio);
    for (int i = 0; i < size; i++)
        x_[i] = x_start[i] + linesearch.x() * p[i];
}

void BFGS::update_inv_hess(const std::vector<double>& g) {
    for (int i = 0; i < size; i++)
        delta_grad[i] = g[i] - g_start[i];

    const double alpha = linesearch.x();
    double py = 0.0;
    for (int i = 0; i < size; i++)
        py += alpha * p[i] * delta_grad[i];

    // the very first iteration
    if (py == 0) return;

    double yCy = 0.0;
    std::fill(yC.begin(), yC.end(), 0.0);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++)
            yC[i] += inv_hess[(size_t) i * size + j] * delta_grad[j];
        yCy += yC[i] * delta_grad[i];
    }

    const double foo = (py + yCy) / (py * py);
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            inv_hess[(size_t) i * size + j] += alpha * alpha * p[i] * p[j] * foo - alpha * (p[i] * yC[j] + yC[i] * p[j]) / py;
}
