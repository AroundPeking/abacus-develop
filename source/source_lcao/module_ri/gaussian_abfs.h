//=======================
// AUTHOR : jiyy
// DATE :   2024-02-27
//=======================

#ifndef GAUSSIAN_ABFS_H
#define GAUSSIAN_ABFS_H

#include <vector>

class Gaussian_Abfs
{
  public:
    /*
    Calculate the lattice sum over a Gaussian:
      S(k) := \sum_G |k+G|^{power+L} \exp(-gamma*|k+G|^2) Y_{LM}(k+G) * \exp(i(k+G)\tau)
    */
    static std::complex<double> lattice_sum(
        const std::vector<ModuleBase::Vector3<double>>& gk,
        const double& power,            // Will be 0. for straight GTOs and -2. for Coulomb interaction
        const double& gamma,
        const bool& exclude_zero, // The R==0. can be excluded by this flag.
        const int& lmax, // Maximum angular momentum the sum is needed for.
        ModuleBase::Vector3<double>& tau);
}

#endif