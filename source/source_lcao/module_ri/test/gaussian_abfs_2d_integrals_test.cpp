#include "../gaussian_abfs_2d_integrals.h"

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace
{
void require(const bool condition, const char* message)
{
    if (!condition)
    {
        std::cerr << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}
} // namespace

int main()
{
    const double coefficient = std::sqrt(3.0 / (4.0 * GaussianAbfs2D::pi));

    const double q_in_plane = 0.06029258;
    const double expected_in_plane
        = -coefficient * GaussianAbfs2D::pi * std::exp(q_in_plane * q_in_plane)
          * std::erfc(q_in_plane);
    require(std::abs(GaussianAbfs2D::coulomb_l1(1, q_in_plane, 0.0, 0.0, 1.0).real()
                     - expected_in_plane)
                < 1.0e-13,
            "in-plane x L=1 analytic Coulomb integral mismatch");
    require(std::abs(GaussianAbfs2D::coulomb_l1(2, 0.0, q_in_plane, 0.0, 1.0).real()
                     - expected_in_plane)
                < 1.0e-13,
            "in-plane y L=1 analytic Coulomb integral mismatch");

    const double q_out_of_plane = 0.10048763;
    const double tau = 0.7;
    const double step = 1.0e-5;
    const double finite_difference
        = (GaussianAbfs2D::i2(q_out_of_plane, tau + step, 1.0)
           - GaussianAbfs2D::i2(q_out_of_plane, tau - step, 1.0))
          / (2.0 * step);
    require(std::abs(GaussianAbfs2D::di2_dtau(q_out_of_plane, tau, 1.0)
                     - finite_difference)
                < 2.0e-9,
            "out-of-plane I2 derivative mismatch");
    require(std::abs(GaussianAbfs2D::coulomb_l1(0, q_out_of_plane, 0.0, tau, 1.0).imag()
                     + coefficient * finite_difference)
                < 2.0e-9,
            "out-of-plane L=1 analytic Coulomb integral mismatch");

    const double small_q_value
        = GaussianAbfs2D::coulomb_l1(1, 1.0e-8, 0.0, 0.0, 1.0).real();
    require(std::abs(small_q_value + coefficient * GaussianAbfs2D::pi) < 2.0e-8,
            "in-plane L=1 small-q limit is not finite");

    const double qx = 0.08;
    const double qy = 0.06;
    const double q = std::hypot(qx, qy);
    const double tau_high_l = 3.7;
    const double i2_value = GaussianAbfs2D::i2(q, tau_high_l, 1.0);
    const double j0 = std::sqrt(GaussianAbfs2D::pi)
                      * std::exp(-tau_high_l * tau_high_l / 4.0);
    const double y20_prefactor = std::sqrt(5.0 / (16.0 * GaussianAbfs2D::pi));
    const std::complex<double> expected_y20
        = y20_prefactor * (2.0 * j0 - 3.0 * q * q * i2_value);
    require(std::abs(GaussianAbfs2D::integral_lm(2, 0, -2.0, qx, qy, tau_high_l, 1.0)
                     - expected_y20)
                < 2.0e-13,
            "L=2 M=0 closed-form Coulomb integral mismatch");

    const double y21_prefactor = std::sqrt(15.0 / (4.0 * GaussianAbfs2D::pi));
    const std::complex<double> expected_y21(
        0.0,
        y21_prefactor * qx * GaussianAbfs2D::di2_dtau(q, tau_high_l, 1.0));
    require(std::abs(GaussianAbfs2D::integral_lm(2, 1, -2.0, qx, qy, tau_high_l, 1.0)
                     - expected_y21)
                < 2.0e-13,
            "L=2 M=1 closed-form Coulomb integral mismatch");

    const double y22_prefactor = std::sqrt(15.0 / (16.0 * GaussianAbfs2D::pi));
    const std::complex<double> expected_y22 = y22_prefactor * (qx * qx - qy * qy) * i2_value;
    require(std::abs(GaussianAbfs2D::integral_lm(2, 3, -2.0, qx, qy, tau_high_l, 1.0)
                     - expected_y22)
                < 2.0e-13,
            "L=2 M=3 closed-form Coulomb integral mismatch");

    const std::complex<double> j2
        = std::sqrt(GaussianAbfs2D::pi) * std::exp(-tau_high_l * tau_high_l / 4.0)
          * (0.5 - tau_high_l * tau_high_l / 4.0);
    const std::complex<double> expected_y20_power0
        = y20_prefactor * (2.0 * j2 - q * q * j0);
    require(std::abs(GaussianAbfs2D::integral_lm(2, 0, 0.0, qx, qy, tau_high_l, 1.0)
                     - expected_y20_power0)
                < 2.0e-13,
            "L=2 M=0 polynomial Gaussian integral mismatch");

    for (const double beta : {0.5, 1.0, 2.0})
    {
        for (int l = 2; l <= 16; ++l)
        {
            const int degree = l - 2;
            const double scaled_tau = tau_high_l / std::sqrt(beta);
            // Physicists' Hermite H_degree at tau_z/(2*sqrt(beta)), the
            // argument of the exact Fourier-Gaussian moment. The previous
            // probabilists' recurrence (He_degree at tau_z/sqrt(beta)) made
            // every degree >= 2 expectation wrong by O(1).
            const double hermite_x = 0.5 * scaled_tau;
            double hm2 = 1.0;
            double hm1 = 2.0 * hermite_x;
            double hermite = degree == 0 ? hm2 : hm1;
            for (int n = 2; n <= degree; ++n)
            {
                hermite = 2.0 * hermite_x * hm1 - 2.0 * (n - 1) * hm2;
                hm2 = hm1;
                hm1 = hermite;
            }
            const std::complex<double> i_power
                = std::polar(1.0, 0.5 * GaussianAbfs2D::pi * static_cast<double>(degree));
            const std::complex<double> exact_gamma
                = std::sqrt(static_cast<double>(2 * l + 1) / (4.0 * GaussianAbfs2D::pi))
                  * std::pow(beta, -0.5 * static_cast<double>(degree + 1))
                  * std::sqrt(GaussianAbfs2D::pi) / std::pow(2.0, degree) * hermite
                  * std::exp(-scaled_tau * scaled_tau / 4.0) * i_power;
            require(std::abs(GaussianAbfs2D::integral_lm(l, 0, -2.0, 0.0, 0.0, tau_high_l, beta)
                             - exact_gamma)
                        // The factorial-ratio solid-harmonic coefficients
                        // accumulate ~3e-11 relative rounding at L=16 (28!
                        // and 32! exceed exact double representation), so a
                        // 1e-10 relative floor is the honest tolerance.
                        < 1.0e-10 * std::max(1.0, std::abs(exact_gamma)),
                    "high-L Gamma closed-form integral mismatch");
        }
    }

    // At q_parallel=0 every M=0 polynomial channel reduces to one exact
    // Fourier-Gaussian moment. This exercises every production power through
    // degree 30 at a nonzero layer separation, where direct quadrature suffers
    // severe oscillatory cancellation.
    for (const double beta : {0.5, 1.0, 2.0})
    {
        for (int power = 0; power <= 14; power += 2)
        {
            for (int l = 0; l <= 16; ++l)
            {
                const int degree = l + power;
                const double scaled_tau = tau_high_l / std::sqrt(beta);
                // Physicists' Hermite H_degree at tau_z/(2*sqrt(beta)); see
                // the note in the high-L loop above.
                const double hermite_x = 0.5 * scaled_tau;
                double hm2 = 1.0;
                double hm1 = 2.0 * hermite_x;
                double hermite = degree == 0 ? hm2 : hm1;
                for (int n = 2; n <= degree; ++n)
                {
                    hermite = 2.0 * hermite_x * hm1 - 2.0 * (n - 1) * hm2;
                    hm2 = hm1;
                    hm1 = hermite;
                }
                const std::complex<double> i_power
                    = std::polar(1.0, 0.5 * GaussianAbfs2D::pi * static_cast<double>(degree));
                const std::complex<double> exact_gamma
                    = std::sqrt(static_cast<double>(2 * l + 1) / (4.0 * GaussianAbfs2D::pi))
                      * std::pow(beta, -0.5 * static_cast<double>(degree + 1))
                      * std::sqrt(GaussianAbfs2D::pi) / std::pow(2.0, degree) * hermite
                      * std::exp(-scaled_tau * scaled_tau / 4.0) * i_power;
                require(std::abs(GaussianAbfs2D::integral_lm(l,
                                                            0,
                                                            static_cast<double>(power),
                                                            0.0,
                                                            0.0,
                                                            tau_high_l,
                                                            beta)
                                 - exact_gamma)
                            < 5.0e-10 * std::max(1.0, std::abs(exact_gamma)),
                        "all-power Gamma Fourier-Gaussian moment mismatch");
            }
        }
    }

    // Exact-Gamma (q_parallel = 0) closed forms for the L=1 Ewald kernel.
    // dI2/dtau_z has a finite Q->0 limit -pi*erf(tau_z/(2*sqrt(beta))), odd in
    // tau_z; the L=1, M=0 kernel K10 = -i*c1*dI2/dtau_z must reproduce it,
    // while the in-plane M=+-1 kernels vanish because they carry a factor Q.
    for (const double beta : {0.5, 1.0, 2.0})
    {
        for (const double tau_z : {-5.9304, -2.9652, 2.9652, 5.9304})
        {
            const double expected
                = -GaussianAbfs2D::pi * std::erf(tau_z / (2.0 * std::sqrt(beta)));
            require(std::abs(GaussianAbfs2D::di2_dtau(0.0, tau_z, beta) - expected) < 1.0e-14,
                    "Q=0 closed-form limit of dI2/dtau_z mismatch");
            const std::complex<double> k10
                = GaussianAbfs2D::coulomb_l1(0, 0.0, 0.0, tau_z, beta);
            require(std::abs(k10.real()) < 1.0e-14,
                    "Q=0 L=1 M=0 kernel must be purely imaginary");
            require(std::abs(k10.imag() + coefficient * expected) < 1.0e-14,
                    "Q=0 L=1 M=0 kernel erf-limit mismatch");
            require(std::abs(GaussianAbfs2D::coulomb_l1(1, 0.0, 0.0, tau_z, beta)) < 1.0e-13
                        && std::abs(GaussianAbfs2D::coulomb_l1(2, 0.0, 0.0, tau_z, beta))
                               < 1.0e-13,
                    "Q=0 L=1 M=+-1 kernels must vanish");
        }
    }
    require(std::abs(GaussianAbfs2D::di2_dtau(0.0, 0.0, 1.0)) < 1.0e-14
                && std::abs(GaussianAbfs2D::coulomb_l1(0, 0.0, 0.0, 0.0, 1.0)) < 1.0e-14,
            "Q=0 L=1 kernels must vanish at tau_z = 0");
    // The closed form must be the continuous extension of the finite-q branch.
    for (const double tau_z : {-5.9304, -2.9652, 2.9652, 5.9304})
    {
        const double closed = GaussianAbfs2D::coulomb_l1(0, 0.0, 0.0, tau_z, 1.0).imag();
        const double small_q = GaussianAbfs2D::coulomb_l1(0, 1.0e-9, 0.0, tau_z, 1.0).imag();
        require(std::abs(closed - small_q) < 1.0e-7,
                "L=1 M=0 kernel discontinuous at Q=0");
    }

    return EXIT_SUCCESS;
}
