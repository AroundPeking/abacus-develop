//=======================
// AUTHOR : jiyy
// DATE :   2024-02-27
//=======================

#ifndef GAUSSIAN_ABFS_CPP
#define GAUSSIAN_ABFS_CPP

#include "gaussian_abfs.h"

#include "module_base/global_variable.h"
#include "module_base/math_ylmreal.h"
#include "module_base/timer.h"
#include "module_base/tool_title.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"

std::complex<double> Gaussian_Abfs::lattice_sum(
    const std::vector<ModuleBase::Vector3<double>>& gk,
    const double& power, // Will be 0. for straight GTOs and -2. for Coulomb interaction
    const double& gamma,
    const bool& exclude_zero, // The R==0. can be excluded by this flag.
    const int& lmax,          // Maximum angular momentum the sum is needed for.
    ModuleBase::Vector3<double>& tau);
{
    ModuleBase::TITLE("Gaussian_Abfs", "lattice_sum");
    ModuleBase::timer::tick("Gaussian_Abfs", "lattice_sum");
    if (power < 0.0 && !exclude_zero)
        ModuleBase::WARNING_QUIT("Gaussian_Abfs::lattice_sum", "Gamma point for power<0.0 cannot be evaluated!");

    const int npw = gk.size();
    const int total_lm = (lmax + 1) * (lmax + 1);
    ModuleBase::matrix ylm(total_lm, npw);
    ModuleBase::YlmReal::Ylm_Real(total_lm, npw, gk.data(), ylm);

    std::complex<double> result(0.0, 0.0);
    for (size_t ig = 0; ig != npw; ++ig)
    {
        ModuleBase::Vector3<double> gk_vec = gk[ig];
        if (exclude_zero && gk_vec.x < 1e-10 && gk_vec.y < 1e-10 && gk_vec.z < 1e-10)
            continue;
        std::complex<double> phase = std::exp(ModuleBase::IMAG_UNIT * (gk_vec * tau));
        std::complex<double> val_s = std::exp(-gamma * gk_vec.norm2()) * std::pow(gk_vec.norm(), power) * phase;

        for (size_t L = 0; L!= lmax; ++L)
        {
            val_s *= std::pow(gk_vec.norm(), L);
            for (size_t m = 0; m != 2 * L + 1; ++m)
            {
                const int lm = L * L + m;
                val_s *= ylm(lm, ig);
            }
        }

        result += val_s;
    }

    ModuleBase::timer::tick("Gaussian_Abfs", "lattice_sum");
    return val_s;
}

#endif