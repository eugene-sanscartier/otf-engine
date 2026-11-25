/* -*- c++ -*- ----------------------------------------------------------
   Real spherical harmonics and their 3-j symbols.
------------------------------------------------------------------------- */

#include "etn_harmonics.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <stdexcept>

RealHarmonics::RealHarmonics(int l_count) : l_count(l_count) {
    if (l_count < 1)
        throw std::runtime_error("RealHarmonics: l_count must be positive");

    vals.resize(l_count * l_count);
    grads.resize(l_count * l_count * 3);
    legendre.assign(l_count * (l_count + 1), 0.0);
    cos_m.resize(l_count);
    sin_m.resize(l_count);

    // N_lm = sqrt((2l+1)/(4 pi) (l-|m|)!/(l+|m|)!), times sqrt(2) for m != 0
    norms.resize(l_count * l_count);
    for (int l = 0; l < l_count; l++)
        for (int m = -l; m <= l; m++) {
            const int k = std::abs(m);
            double ratio = 1.0;
            for (int f = l - k + 1; f <= l + k; f++) ratio /= f;
            norms[l * l + l + m] = std::sqrt((2 * l + 1) / (4 * M_PI) * ratio) * (k == 0 ? 1.0 : std::sqrt(2.0));
        }
}

// Y_l0 = N P_l(z), Y_lm = N d^m P_l/dz^m Re (x + i y)^m, Y_l,-m = N d^m P_l/dz^m Im (x + i y)^m.
void RealHarmonics::compute(const double u[3]) {
    const double x = u[0], y = u[1], z = u[2];
    const int stride = l_count + 1;

    cos_m[0] = 1.0;
    sin_m[0] = 0.0;
    for (int m = 1; m < l_count; m++) {
        cos_m[m] = x * cos_m[m - 1] - y * sin_m[m - 1];
        sin_m[m] = x * sin_m[m - 1] + y * cos_m[m - 1];
    }

    // d^m P_l / dz^m by the recursion in l at fixed m, from (2m-1)!! at l = m
    double diagonal = 1.0;
    for (int m = 0; m < l_count; m++) {
        if (m > 0) diagonal *= 2 * m - 1;
        legendre[m * stride + m] = diagonal;
        if (m + 1 < l_count) legendre[(m + 1) * stride + m] = (2 * m + 1) * z * diagonal;
        for (int l = m + 2; l < l_count; l++)
            legendre[l * stride + m] = ((2 * l - 1) * z * legendre[(l - 1) * stride + m] - (l + m - 1) * legendre[(l - 2) * stride + m]) / (l - m);
    }

    for (int l = 0; l < l_count; l++)
        for (int m = -l; m <= l; m++) {
            const int k = std::abs(m);
            const int index = l * l + l + m;
            const double norm = norms[index];
            const double p = legendre[l * stride + k];
            const double dp = legendre[l * stride + k + 1];  // d/dz of p; zero when k = l
            double* g = &grads[index * 3];

            if (m >= 0) {
                vals[index] = norm * p * cos_m[k];
                g[0] = k > 0 ? norm * p * k * cos_m[k - 1] : 0.0;
                g[1] = k > 0 ? -norm * p * k * sin_m[k - 1] : 0.0;
                g[2] = norm * dp * cos_m[k];
            } else {
                vals[index] = norm * p * sin_m[k];
                g[0] = norm * p * k * sin_m[k - 1];
                g[1] = norm * p * k * cos_m[k - 1];
                g[2] = norm * dp * sin_m[k];
            }

            const double radial = g[0] * x + g[1] * y + g[2] * z;
            g[0] -= radial * x;
            g[1] -= radial * y;
            g[2] -= radial * z;
        }
}

void RealHarmonics::compute_rows(int count, const double* x, const double* y, const double* z) {
    const int stride = l_count + 1;
    row_count = count;
    rows_vals.resize(vals.size() * count);
    rows_grads.resize(grads.size() * count);
    double* vals_rows = rows_vals.data();
    double* grads_rows = rows_grads.data();
    rows_cos.resize((size_t) l_count * count);
    rows_sin.resize((size_t) l_count * count);
    rows_legendre.assign((size_t) l_count * stride * count, 0.0);

    for (int j = 0; j < count; j++) {
        rows_cos[j] = 1.0;
        rows_sin[j] = 0.0;
    }
    for (int m = 1; m < l_count; m++) {
        const double* c0 = rows_cos.data() + (size_t) (m - 1) * count;
        const double* s0 = rows_sin.data() + (size_t) (m - 1) * count;
        double* c1 = rows_cos.data() + (size_t) m * count;
        double* s1 = rows_sin.data() + (size_t) m * count;
        for (int j = 0; j < count; j++) {
            c1[j] = x[j] * c0[j] - y[j] * s0[j];
            s1[j] = x[j] * s0[j] + y[j] * c0[j];
        }
    }

    double diagonal = 1.0;
    for (int m = 0; m < l_count; m++) {
        if (m > 0) diagonal *= 2 * m - 1;
        double* pmm = rows_legendre.data() + (size_t) (m * stride + m) * count;
        for (int j = 0; j < count; j++) pmm[j] = diagonal;
        if (m + 1 < l_count) {
            double* p1 = rows_legendre.data() + (size_t) ((m + 1) * stride + m) * count;
            for (int j = 0; j < count; j++) p1[j] = (2 * m + 1) * z[j] * diagonal;
        }
        for (int l = m + 2; l < l_count; l++) {
            const double* pa = rows_legendre.data() + (size_t) ((l - 1) * stride + m) * count;
            const double* pb = rows_legendre.data() + (size_t) ((l - 2) * stride + m) * count;
            double* pl = rows_legendre.data() + (size_t) (l * stride + m) * count;
            for (int j = 0; j < count; j++) pl[j] = ((2 * l - 1) * z[j] * pa[j] - (l + m - 1) * pb[j]) / (l - m);
        }
    }

    for (int l = 0; l < l_count; l++)
        for (int m = -l; m <= l; m++) {
            const int k = std::abs(m);
            const int index = l * l + l + m;
            const double norm = norms[index];
            const double* p = rows_legendre.data() + (size_t) (l * stride + k) * count;
            const double* dp = rows_legendre.data() + (size_t) (l * stride + k + 1) * count;
            const double* ck = rows_cos.data() + (size_t) k * count;
            const double* sk = rows_sin.data() + (size_t) k * count;
            const double* ck1 = rows_cos.data() + (size_t) (k > 0 ? k - 1 : 0) * count;
            const double* sk1 = rows_sin.data() + (size_t) (k > 0 ? k - 1 : 0) * count;
            double* v = vals_rows + (size_t) index * count;
            double* g0 = grads_rows + (size_t) (index * 3 + 0) * count;
            double* g1 = grads_rows + (size_t) (index * 3 + 1) * count;
            double* g2 = grads_rows + (size_t) (index * 3 + 2) * count;
            for (int j = 0; j < count; j++) {
                double a, b, c;
                if (m >= 0) {
                    v[j] = norm * p[j] * ck[j];
                    a = k > 0 ? norm * p[j] * k * ck1[j] : 0.0;
                    b = k > 0 ? -norm * p[j] * k * sk1[j] : 0.0;
                    c = norm * dp[j] * ck[j];
                } else {
                    v[j] = norm * p[j] * sk[j];
                    a = norm * p[j] * k * sk1[j];
                    b = norm * p[j] * k * ck1[j];
                    c = norm * dp[j] * sk[j];
                }
                const double radial = a * x[j] + b * y[j] + c * z[j];
                g0[j] = a - radial * x[j];
                g1[j] = b - radial * y[j];
                g2[j] = c - radial * z[j];
            }
        }
}

// Racah's formula in long double.
double wigner_3j(int l1, int l2, int l3, int m1, int m2, int m3) {
    if (m1 + m2 + m3 != 0 || std::abs(m1) > l1 || std::abs(m2) > l2 || std::abs(m3) > l3) return 0.0;
    if (2 * std::max({l1, l2, l3}) > l1 + l2 + l3) return 0.0;

    auto factorial = [](int n) {
        long double f = 1.0L;
        for (int k = 2; k <= n; k++) f *= k;
        return f;
    };

    const long double triangle = factorial(l1 + l2 - l3) * factorial(l1 - l2 + l3) * factorial(-l1 + l2 + l3) / factorial(l1 + l2 + l3 + 1);
    const long double prefactor = std::sqrt(triangle * factorial(l1 + m1) * factorial(l1 - m1) * factorial(l2 + m2) * factorial(l2 - m2) * factorial(l3 + m3) * factorial(l3 - m3));

    const int k_min = std::max({0, l2 - l3 - m1, l1 - l3 + m2});
    const int k_max = std::min({l1 + l2 - l3, l1 - m1, l2 + m2});
    long double sum = 0.0L;
    for (int k = k_min; k <= k_max; k++) {
        const long double term = 1.0L / (factorial(k) * factorial(l3 - l2 + k + m1) * factorial(l3 - l1 + k - m2) * factorial(l1 + l2 - l3 - k) * factorial(l1 - k - m1) * factorial(l2 - k + m2));
        sum += (k % 2 == 0) ? term : -term;
    }

    const int phase = l1 - l2 - m3;
    const long double value = ((phase % 2 == 0) ? 1.0L : -1.0L) * prefactor * sum;
    return (double) value;
}

// Entry (m, mp) of the matrix U that carries complex harmonic mp to real harmonic m.
static std::complex<double> to_real(int m, int mp) {
    const double half = std::sqrt(0.5);
    if (m < 0) {
        if (mp == m) return {0.0, half};
        if (mp == -m) return {0.0, (mp % 2 == 0) ? -half : half};
        return 0.0;
    }
    if (m == 0) return mp == 0 ? 1.0 : 0.0;
    if (mp == m) return (mp % 2 == 0) ? half : -half;
    if (mp == -m) return half;
    return 0.0;
}

// Sum over the complex m' = +-m of the 3-j symbol times conj(U(m, m')) for each harmonic.
double real_wigner_3j(int l1, int m1, int l2, int m2, int l3, int m3) {
    if (2 * std::max({l1, l2, l3}) > l1 + l2 + l3) return 0.0;

    std::complex<double> sum = 0.0;
    for (int s1 : {1, -1}) {
        if (m1 == 0 && s1 < 0) continue;
        for (int s2 : {1, -1}) {
            if (m2 == 0 && s2 < 0) continue;
            for (int s3 : {1, -1}) {
                if (m3 == 0 && s3 < 0) continue;
                const int p1 = s1 * m1, p2 = s2 * m2, p3 = s3 * m3;
                const double w = wigner_3j(l1, l2, l3, p1, p2, p3);
                if (w == 0.0) continue;
                sum += w * std::conj(to_real(m1, p1)) * std::conj(to_real(m2, p2)) * std::conj(to_real(m3, p3));
            }
        }
    }

    return ((l1 + l2 + l3) % 2 == 0) ? sum.real() : sum.imag();
}

// Keeps the symbols above 1e-15 in magnitude; the others are zeros up to rounding.
std::vector<WignerEntry> real_wigner_entries(int l1, int l2, int l3) {
    std::vector<WignerEntry> entries;
    for (int m3 = -l3; m3 <= l3; m3++)
        for (int m2 = -l2; m2 <= l2; m2++)
            for (int m1 = -l1; m1 <= l1; m1++) {
                const double w = real_wigner_3j(l1, m1, l2, m2, l3, m3);
                if (std::abs(w) > 1e-15) entries.push_back({m1, m2, m3, w});
            }
    return entries;
}
